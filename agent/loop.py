"""
V2 agent loop. Platform-agnostic: works with OSNativePlatform (desktop),
OSWorldPlatform (VM), or any other concrete Platform.

Only three V1 → V2 signature changes:
  - `browser_page` -> `platform: Platform`
  - `start_url` removed (platform.setup() handles initial state)
  - no more `async_playwright` context — caller does `async with platform`
"""

import asyncio
import logging

from vision.diff import compute_diff, extract_crops
from vision.classifier import classify_transition, extract_anchor, TransitionType
from observation.builder import build_observation
from agent.state import AgentState
from capture.base import Platform
from safety import SafetyLayer

logger = logging.getLogger(__name__)


async def run_agent(
    task: str,
    model,
    platform: Platform,
    config,
    safety: SafetyLayer | None = None,
) -> AgentState:
    """
    Main DeltaVision-OS agent loop.

    Terminates when:
      - model returns done=True
      - max_steps exceeded
      - max_consecutive_failures exceeded
    """
    state = AgentState(task=task)

    # Platform has already been set up by the caller (async with platform:).
    t0 = await platform.capture()
    url_t0 = await platform.get_url()
    a11y_t0 = await platform.get_a11y_xml()
    anchor_template = extract_anchor(t0, config)

    obs = build_observation(
        obs_type="full_frame",
        task=task,
        step=0,
        last_action=None,
        frame=t0,
        url=url_t0 or "",
        trigger_reason="initial",
        a11y_xml=a11y_t0,
    )
    state.add_observation(obs)

    while not state.done and state.step < config.MAX_STEPS:
        # Model predicts next action
        response = await model.predict(obs, state)
        state.add_response(response)

        if response.action is None or response.done:
            state.done = True
            logger.info("Agent finished at step %d. Reason: %s", state.step, response.reasoning)
            break

        action = response.action
        logger.info("Step %d: %s (confidence=%.2f)", state.step, action, response.confidence)

        # Safety check — runs regardless of model backend
        if safety is not None:
            url_now = await platform.get_url()
            check = safety.check_action(action, url_now or "")
            if not check.allowed:
                logger.warning("SAFETY BLOCK: %s", check.reason)
                state.step += 1
                frame_now = await platform.capture()
                obs = build_observation(
                    obs_type="full_frame",
                    task=task,
                    step=state.step,
                    last_action=action,
                    frame=frame_now,
                    url=(await platform.get_url()) or "",
                    trigger_reason=f"safety_block:{check.reason}",
                    a11y_xml=await platform.get_a11y_xml(),
                )
                state.add_observation(obs)
                continue

        # Execute via the platform
        url_before = await platform.get_url()
        await platform.execute(action)

        # Wait for the environment to react
        await asyncio.sleep(config.POST_ACTION_WAIT_MS / 1000)

        # Capture and classify
        t1 = await platform.capture()
        url_after = await platform.get_url()
        a11y_after = await platform.get_a11y_xml()

        diff_result = compute_diff(t0, t1, config)
        classification = classify_transition(
            t0=t0,
            t1=t1,
            url_before=url_before or "",
            url_after=url_after or "",
            anchor_template=anchor_template,
            config=config,
            diff_result=diff_result,
            last_action_type=action.type.value,
        )
        state.log_transition(classification, action, state.step)

        logger.debug(
            "Transition: %s (trigger=%s, diff=%.3f, phash=%d, anchor=%.2f)",
            classification.transition.value,
            classification.trigger,
            classification.diff_ratio,
            classification.phash_distance,
            classification.anchor_score,
        )

        force_full = getattr(config, "FORCE_FULL_FRAME", False)

        if classification.transition == TransitionType.NEW_PAGE or force_full:
            t0 = t1
            url_t0 = url_after
            anchor_template = extract_anchor(t0, config)

            if classification.transition == TransitionType.NEW_PAGE:
                state.reset_no_change_streak()
                state.increment_new_page_count()

            trigger = classification.trigger if not force_full else f"forced_full|{classification.trigger}"
            obs = build_observation(
                obs_type="full_frame",
                task=task,
                step=state.step,
                last_action=action,
                frame=t1,
                url=url_after or "",
                trigger_reason=trigger,
                a11y_xml=a11y_after,
            )
        else:  # DELTA
            # Re-anchor after scroll since viewport shifted
            from agent.actions import ActionType
            if action.type == ActionType.SCROLL:
                t0 = t1
                anchor_template = extract_anchor(t0, config)

            crops = extract_crops(t0, t1, diff_result.changed_bboxes, config.CROP_PADDING)

            if not diff_result.action_had_effect:
                state.increment_no_change_streak()
            else:
                state.reset_no_change_streak()

            if state.no_change_streak >= config.MAX_NO_EFFECT_RETRIES:
                logger.warning(
                    "No-effect streak hit %d — forcing full frame refresh",
                    state.no_change_streak,
                )
                t0_refresh = await platform.capture()
                a11y_refresh = await platform.get_a11y_xml()
                obs = build_observation(
                    obs_type="full_frame",
                    task=task,
                    step=state.step,
                    last_action=action,
                    frame=t0_refresh,
                    url=url_after or "",
                    trigger_reason="force_refresh_no_effect",
                    a11y_xml=a11y_refresh,
                )
                state.reset_no_change_streak()
                t0 = t0_refresh
                anchor_template = extract_anchor(t0, config)
            else:
                obs = build_observation(
                    obs_type="delta",
                    task=task,
                    step=state.step,
                    last_action=action,
                    diff_result=diff_result,
                    crops=crops,
                    action_had_effect=diff_result.action_had_effect,
                    no_change_count=state.no_change_streak,
                    current_frame=t1,
                    a11y_xml=a11y_after,
                )

        state.add_observation(obs)
        state.step += 1

    logger.info(
        "Run complete. Steps: %d, Delta ratio: %.1f%%, New pages: %d",
        state.step, state.delta_ratio * 100, state.new_page_count,
    )
    return state
