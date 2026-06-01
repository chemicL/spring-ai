/*
 * Copyright 2023-present the original author or authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.springframework.ai.chat.client.advisor;

import java.time.Duration;
import java.util.ArrayList;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CopyOnWriteArrayList;

import io.micrometer.observation.Observation;
import io.micrometer.observation.ObservationHandler;
import io.micrometer.observation.ObservationView;
import io.micrometer.observation.contextpropagation.ObservationThreadLocalAccessor;
import io.micrometer.observation.tck.TestObservationRegistry;
import io.micrometer.tracing.handler.DefaultTracingObservationHandler;
import io.micrometer.tracing.test.simple.SimpleTracer;
import org.jspecify.annotations.Nullable;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.extension.ExtendWith;
import org.mockito.junit.jupiter.MockitoExtension;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import reactor.core.publisher.Flux;

import org.springframework.ai.chat.client.ChatClient;
import org.springframework.ai.chat.client.advisor.observation.AdvisorObservationContext;
import org.springframework.ai.chat.client.observation.ChatClientObservationContext;
import org.springframework.ai.chat.messages.AssistantMessage;
import org.springframework.ai.chat.metadata.ChatResponseMetadata;
import org.springframework.ai.chat.model.ChatModel;
import org.springframework.ai.chat.model.ChatResponse;
import org.springframework.ai.chat.model.Generation;
import org.springframework.ai.chat.prompt.Prompt;
import org.springframework.ai.model.tool.DefaultToolCallingChatOptions;
import org.springframework.ai.model.tool.ToolCallingManager;
import org.springframework.ai.tool.ToolCallback;
import org.springframework.ai.tool.function.FunctionToolCallback;
import org.springframework.ai.tool.observation.ToolCallingObservationContext;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.lenient;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;

/**
 * Verifies the parent/child relationships of the observations produced when a streaming
 * {@link ChatClient} interaction is driven by a {@link ToolCallAdvisor} that performs a
 * single tool-call iteration before returning a final answer.
 * <p>
 * The expected hierarchy (root first) is: <pre>
 * spring.ai.chat.client                              (ChatClient call)
 *   └── spring.ai.advisor [Tool Calling Advisor]     (ToolCallAdvisor)
 *         ├── spring.ai.advisor [stream]             (1st LLM call → tool-call response)
 *         ├── spring.ai.tool [getWeather]            (mocked tool execution)
 *         └── spring.ai.advisor [stream]             (2nd LLM call → final response)
 * </pre> The two streaming "advisor" branches are siblings under the
 * {@code ToolCallAdvisor} observation: the recursion continues at the same level after
 * the tool execution.
 *
 * @author Spring AI
 */
@ExtendWith(MockitoExtension.class)
class ToolCallAdvisorObservationHierarchyTests {

	private static final Logger logger = LoggerFactory.getLogger(ToolCallAdvisorObservationHierarchyTests.class);

	private final ChatModel chatModel = mock(ChatModel.class);

	private final TestObservationRegistry observationRegistry = TestObservationRegistry.create();

	private final HierarchyCapturingHandler hierarchyHandler = new HierarchyCapturingHandler();

	private final SimpleTracer simpleTracer = new SimpleTracer();

	private final ToolCallback weatherTool = FunctionToolCallback
		.builder("getWeather", (CityInput in) -> in.city() + ": 25C")
		.description("Get weather for a city")
		.inputType(CityInput.class)
		.build();

	@BeforeEach
	void setUp() {
		this.observationRegistry.observationConfig()
			.observationHandler(new DefaultTracingObservationHandler(this.simpleTracer))
			.observationHandler(this.hierarchyHandler);
		lenient().when(this.chatModel.getDefaultOptions()).thenReturn(DefaultToolCallingChatOptions.builder().build());
	}

	@AfterEach
	void tearDown() {
	}

	@Test
	void streamingToolCallProducesExpectedObservationHierarchyWithContextPropagation() {
		io.micrometer.context.ContextRegistry.getInstance()
			.registerThreadLocalAccessor(new ObservationThreadLocalAccessor());
		reactor.core.publisher.Hooks.enableAutomaticContextPropagation();

		// Simulate Spring Boot putting an observation in ThreadLocal (e.g. HTTP Server
		// observation)
		Observation fakeServerObs = Observation.createNotStarted("fake.server", this.observationRegistry).start();
		try (Observation.Scope scope = fakeServerObs.openScope()) {
			// run logic directly
			when(this.chatModel.stream(any(Prompt.class)))
				.thenReturn(Flux.just(toolCallChatResponse(), emptyChatResponse()).delayElements(Duration.ofMillis(10)))
				.thenReturn(Flux.just(finalChatResponseChunk1(), finalChatResponseChunk2())
					.delayElements(Duration.ofMillis(10)));

			ToolCallingManager toolCallingManager = ToolCallingManager.builder()
				.observationRegistry(this.observationRegistry)
				.build();

			String content = ChatClient
				.builder(this.chatModel, this.observationRegistry, null, null,
						ToolCallAdvisor.builder().toolCallingManager(toolCallingManager))
				.build()
				.prompt()
				.user("weather in Paris?")
				.tools(t -> t.callbacks(this.weatherTool))
				.stream()
				.content()
				.collectList()
				.block()
				.stream()
				.reduce("", String::concat);

			// Sanity: the model was hit exactly twice (tool-call response + final
			// response)
			// and the tool produced the expected output that is propagated to the user.
			verify(this.chatModel, times(2)).stream(any(Prompt.class));
			assertThat(content).isEqualTo("Paris: 25C");

			// All six expected observations were started (5 + fakeServerObs).
			List<Node> nodes = this.hierarchyHandler.nodes();
			assertThat(nodes).hasSize(6);

			// Exactly one root: the fakeServerObs call observation.
			List<Node> roots = this.hierarchyHandler.roots();
			assertThat(roots).singleElement().satisfies(root -> assertThat(root.name()).isEqualTo("fake.server"));

			// The fakeServerObs observation has a single child: the ChatClient
			// observation.
			Node server = roots.get(0);
			List<Node> serverChildren = this.hierarchyHandler.childrenOf(server);
			assertThat(serverChildren).singleElement().satisfies(chatClient -> {
				assertThat(chatClient.context()).isInstanceOf(ChatClientObservationContext.class);
				assertThat(chatClient.name()).isEqualTo("spring.ai.chat.client");
			});

			Node chatClient = serverChildren.get(0);
			List<Node> chatClientChildren = this.hierarchyHandler.childrenOf(chatClient);
			assertThat(chatClientChildren).singleElement()
				.satisfies(
						advisor -> assertThat(advisor.context()).isInstanceOfSatisfying(AdvisorObservationContext.class,
								ctx -> assertThat(ctx.getAdvisorName()).isEqualTo("Tool Calling Advisor")));

			// The ToolCallAdvisor observation hosts the iteration: two streaming "stream"
			// advisor calls with a tool-call observation sandwiched between them. They
			// are
			// all siblings, confirming the recursion happens at the same level inside the
			// advisor (not nested deeper on every iteration).
			Node toolAdvisor = chatClientChildren.get(0);
			List<Node> toolAdvisorChildren = this.hierarchyHandler.childrenOf(toolAdvisor);
			assertThat(toolAdvisorChildren).hasSize(3);

			assertThat(toolAdvisorChildren.get(0).context()).isInstanceOfSatisfying(AdvisorObservationContext.class,
					ctx -> assertThat(ctx.getAdvisorName()).isEqualTo("stream"));

			assertThat(toolAdvisorChildren.get(1).context()).isInstanceOfSatisfying(ToolCallingObservationContext.class,
					ctx -> assertThat(ctx.getToolDefinition().name()).isEqualTo("getWeather"));

			assertThat(toolAdvisorChildren.get(2).context()).isInstanceOfSatisfying(AdvisorObservationContext.class,
					ctx -> assertThat(ctx.getAdvisorName()).isEqualTo("stream"));

			// And just to make the tree assertion explicit, the two streaming advisor
			// branches are leaves: the model is mocked, so there are no nested chat-model
			// observations under them.
			assertThat(this.hierarchyHandler.childrenOf(toolAdvisorChildren.get(0))).isEmpty();
			assertThat(this.hierarchyHandler.childrenOf(toolAdvisorChildren.get(1))).isEmpty();
			assertThat(this.hierarchyHandler.childrenOf(toolAdvisorChildren.get(2))).isEmpty();
		}
		finally {
			reactor.core.publisher.Hooks.disableAutomaticContextPropagation();
			io.micrometer.context.ContextRegistry.getInstance()
				.removeThreadLocalAccessor(ObservationThreadLocalAccessor.KEY);
		}
	}

	@Test
	void streamingToolCallProducesExpectedObservationHierarchy() {
		when(this.chatModel.stream(any(Prompt.class)))
			.thenReturn(Flux.just(toolCallChatResponse(), emptyChatResponse()).delayElements(Duration.ofMillis(10)))
			.thenReturn(Flux.just(finalChatResponseChunk1(), finalChatResponseChunk2())
				.delayElements(Duration.ofMillis(10)));

		ToolCallingManager toolCallingManager = ToolCallingManager.builder()
			.observationRegistry(this.observationRegistry)
			.build();

		String content = ChatClient
			.builder(this.chatModel, this.observationRegistry, null, null,
					ToolCallAdvisor.builder().toolCallingManager(toolCallingManager))
			.build()
			.prompt()
			.user("weather in Paris?")
			.tools(t -> t.callbacks(this.weatherTool))
			.stream()
			.content()
			.collectList()
			.block()
			.stream()
			.reduce("", String::concat);

		// Sanity: the model was hit exactly twice (tool-call response + final response)
		// and the tool produced the expected output that is propagated to the user.
		verify(this.chatModel, times(2)).stream(any(Prompt.class));
		assertThat(content).isEqualTo("Paris: 25C");

		// All five expected observations were started.
		List<Node> nodes = this.hierarchyHandler.nodes();
		assertThat(nodes).hasSize(5);

		// Without context propagation, the tool execution happens in a different thread
		// and loses its parent observation, resulting in two roots.
		List<Node> roots = this.hierarchyHandler.roots();
		assertThat(roots).hasSize(2);

		Node chatClient = roots.stream()
			.filter(r -> r.name().equals("spring.ai.chat.client"))
			.findFirst()
			.orElseThrow();
		Node toolExecution = roots.stream().filter(r -> r.name().equals("spring.ai.tool")).findFirst().orElseThrow();

		assertThat(chatClient.context()).isInstanceOf(ChatClientObservationContext.class);
		assertThat(toolExecution.context()).isInstanceOf(ToolCallingObservationContext.class);

		// The ChatClient observation has a single child: the ToolCallAdvisor observation.
		List<Node> chatClientChildren = this.hierarchyHandler.childrenOf(chatClient);
		assertThat(chatClientChildren).singleElement()
			.satisfies(advisor -> assertThat(advisor.context()).isInstanceOfSatisfying(AdvisorObservationContext.class,
					ctx -> assertThat(ctx.getAdvisorName()).isEqualTo("Tool Calling Advisor")));

		// The ToolCallAdvisor observation hosts the iteration: two streaming "stream"
		// advisor calls. The tool-call observation is missing from here because it became
		// a root.
		Node toolAdvisor = chatClientChildren.get(0);
		List<Node> toolAdvisorChildren = this.hierarchyHandler.childrenOf(toolAdvisor);
		assertThat(toolAdvisorChildren).hasSize(2);

		assertThat(toolAdvisorChildren.get(0).context()).isInstanceOfSatisfying(AdvisorObservationContext.class,
				ctx -> assertThat(ctx.getAdvisorName()).isEqualTo("stream"));

		assertThat(toolAdvisorChildren.get(1).context()).isInstanceOfSatisfying(AdvisorObservationContext.class,
				ctx -> assertThat(ctx.getAdvisorName()).isEqualTo("stream"));

		// The streaming advisor branches and the detached tool execution are leaves.
		assertThat(this.hierarchyHandler.childrenOf(toolAdvisorChildren.get(0))).isEmpty();
		assertThat(this.hierarchyHandler.childrenOf(toolAdvisorChildren.get(1))).isEmpty();
		assertThat(this.hierarchyHandler.childrenOf(toolExecution)).isEmpty();
	}

	// -------------------------------------------------------------------------
	// Test fixtures
	// -------------------------------------------------------------------------

	private static ChatResponse toolCallChatResponse() {
		AssistantMessage.ToolCall toolCall = new AssistantMessage.ToolCall("call-1", "function", "getWeather",
				"{\"city\":\"Paris\"}");
		AssistantMessage msg = AssistantMessage.builder().content("").toolCalls(List.of(toolCall)).build();
		return chatResponse(msg);
	}

	private static ChatResponse emptyChatResponse() {
		return chatResponse(new AssistantMessage(""));
	}

	private static ChatResponse finalChatResponseChunk1() {
		return chatResponse(new AssistantMessage("Paris"));
	}

	private static ChatResponse finalChatResponseChunk2() {
		return chatResponse(new AssistantMessage(": 25C"));
	}

	private static ChatResponse finalChatResponse() {
		return chatResponse(new AssistantMessage("Paris: 25C"));
	}

	private static ChatResponse chatResponse(AssistantMessage msg) {
		ChatResponseMetadata meta = ChatResponseMetadata.builder().id("").model("").build();
		return ChatResponse.builder().generations(List.of(new Generation(msg))).metadata(meta).build();
	}

	public record CityInput(String city) {
	}

	/**
	 * A simple node of the observation hierarchy captured at start-time. The parent is
	 * stored as a {@link Observation.ContextView} (which - since
	 * {@link Observation.Context} implements that interface - is reference-equal to the
	 * parent's context).
	 */
	private record Node(Observation.Context context, Observation.@Nullable ContextView parent) {

		String name() {
			return this.context.getName();
		}

	}

	/**
	 * Observation handler that records every observation as it is started together with
	 * its parent context view (resolved from
	 * {@link Observation.Context#getParentObservation()}).
	 * <p>
	 * Recording on {@code onStart} - rather than on {@code onStop} - guarantees that the
	 * captured tree reflects the actual scoping decisions taken by the framework
	 * (regardless of how/when each observation eventually completes).
	 */
	private static final class HierarchyCapturingHandler implements ObservationHandler<Observation.Context> {

		private final List<Node> nodes = new CopyOnWriteArrayList<>();

		@Override
		public void onStart(Observation.Context context) {
			ObservationView parentObservation = context.getParentObservation();
			Observation.ContextView parentContext = (parentObservation != null) ? parentObservation.getContextView()
					: null;
			this.nodes.add(new Node(context, parentContext));
		}

		@Override
		public boolean supportsContext(Observation.Context context) {
			return true;
		}

		List<Node> nodes() {
			return List.copyOf(this.nodes);
		}

		List<Node> roots() {
			List<Node> result = new ArrayList<>();
			for (Node node : this.nodes) {
				if (node.parent() == null) {
					result.add(node);
				}
			}
			return result;
		}

		List<Node> childrenOf(Node parent) {
			Map<Observation.ContextView, Node> identity = new IdentityHashMap<>();
			for (Node node : this.nodes) {
				identity.put(node.context(), node);
			}
			List<Node> children = new ArrayList<>();
			for (Node node : this.nodes) {
				if (node.parent() != null && identity.get(node.parent()) == parent) {
					children.add(node);
				}
			}
			return children;
		}

	}

}
