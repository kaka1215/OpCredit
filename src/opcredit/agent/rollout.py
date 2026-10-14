from __future__ import annotations

import asyncio
import time
from functools import lru_cache
from uuid import uuid4

from verl.experimental.agent_loop.agent_loop import (
    AgentLoopBase,
    AgentLoopMetrics,
    AgentLoopOutput,
)

from opcredit.agent.environment import EnvironmentConfig, InvalidOperation, NavigationEnvironment
from opcredit.agent.kg import FreebaseGraph, SemanticRanker
from opcredit.agent.modeling import ChatBuffer
from opcredit.credit.ftc import AnswerNeighborhood
from opcredit.data.prepare import Question
from opcredit.records import DecisionRecorder
from opcredit.training.reward import task_reward


@lru_cache(maxsize=4)
def resources(endpoint, embedding_model, query_timeout):
    return FreebaseGraph(endpoint, query_timeout), SemanticRanker(embedding_model, "cpu")


class OpCreditAgentLoop(AgentLoopBase):
    async def run(self, sampling_params, **kwargs):
        settings = self.config.opcredit
        graph, ranker = resources(
            settings.endpoint, settings.embedding_model, settings.query_timeout
        )
        question = Question.from_dict(kwargs["extra_info"]["question_record"])
        env = NavigationEnvironment(
            graph,
            ranker,
            question.question,
            question.topic_entity,
            EnvironmentConfig(
                settings.max_hops,
                settings.relation_candidates,
                settings.entity_candidates,
                settings.neighbor_limit,
                settings.mediator_predicate_limit,
            ),
        )
        chat = ChatBuffer(self.tokenizer, env.initial_prompt)
        prompt_length = len(chat.input_ids)
        if prompt_length > self.rollout_config.prompt_length:
            raise ValueError("KG prompt exceeds the configured prompt length.")
        recorder = DecisionRecorder()
        invalid_count = 0
        turns = 0
        generation_time = 0.0
        tool_time = 0.0
        response_limit = self.rollout_config.response_length
        request_id = uuid4().hex
        while not env.done:
            remaining = response_limit - (len(chat.input_ids) - prompt_length)
            if remaining <= 0:
                break
            state = env.state
            params = dict(sampling_params)
            allowance = min(settings.max_turn_tokens, remaining)
            params["max_tokens"] = allowance
            start = time.monotonic()
            output = await self.server_manager.generate(
                request_id=request_id, prompt_ids=list(chat.input_ids), sampling_params=params
            )
            generation_time += time.monotonic() - start
            tokens = output.token_ids
            if not tokens or len(tokens) > allowance:
                raise ValueError("The rollout backend returned an invalid token count.")
            text = self.tokenizer.decode(tokens, skip_special_tokens=True)
            chat.append_generation(tokens)
            turns += 1
            start = time.monotonic()
            try:
                operation, frontier, observation = await asyncio.to_thread(env.execute, text)
            except InvalidOperation as error:
                operation, frontier = env.invalid_operation(text), None
                observation = f"Invalid operation: {error} Retry using the last observation."
                invalid_count += 1
            tool_time += time.monotonic() - start
            if frontier is not None and (
                state.active_entity.startswith("lit:")
                or any(entity and entity.startswith("lit:") for entity in frontier)
            ):
                frontier = None
            recorder.record(state, operation, len(tokens), frontier)
            if env.done or invalid_count >= settings.max_invalid_operations:
                break
            retained = len(chat.input_ids)
            chat.append_observation(observation)
            if len(chat.input_ids) - prompt_length >= response_limit:
                del chat.input_ids[retained:]
                del chat.response_mask[retained:]
                break
        reward = task_reward(env.predicted_names, question.answer_names)
        record = recorder.finish(question.question_id, request_id, reward)
        gold = [entity for entity in question.answer_entities if entity.startswith(("m.", "g."))]
        neighborhood = (
            await asyncio.to_thread(
                AnswerNeighborhood.from_graph,
                graph,
                gold,
                settings.ftc_max_predicates,
                settings.ftc_max_neighbors,
            )
            if settings.use_ftc
            else AnswerNeighborhood(frozenset(), frozenset())
        )
        return AgentLoopOutput(
            prompt_ids=chat.input_ids[:prompt_length],
            response_ids=chat.input_ids[prompt_length:],
            response_mask=[int(value) for value in chat.response_mask[prompt_length:]],
            reward_score=reward,
            num_turns=turns,
            metrics=AgentLoopMetrics(generate_sequences=generation_time, tool_calls=tool_time),
            extra_fields={"opcredit_record": record, "opcredit_neighborhood": neighborhood},
        )
