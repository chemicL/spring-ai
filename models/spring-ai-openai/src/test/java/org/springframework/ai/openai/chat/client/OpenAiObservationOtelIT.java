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

package org.springframework.ai.openai.chat.client;

import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.stream.Collectors;

import io.micrometer.observation.Observation;
import io.micrometer.observation.ObservationHandler;
import io.micrometer.observation.ObservationRegistry;
import io.micrometer.observation.tck.TestObservationRegistry;
import io.micrometer.tracing.handler.DefaultTracingObservationHandler;
import io.micrometer.tracing.otel.bridge.OtelBaggageManager;
import io.micrometer.tracing.otel.bridge.OtelCurrentTraceContext;
import io.micrometer.tracing.otel.bridge.OtelTracer;
import io.opentelemetry.api.trace.Tracer;
import io.opentelemetry.sdk.OpenTelemetrySdk;
import io.opentelemetry.sdk.testing.exporter.InMemorySpanExporter;
import io.opentelemetry.sdk.trace.SdkTracerProvider;
import io.opentelemetry.sdk.trace.data.SpanData;
import io.opentelemetry.sdk.trace.export.SimpleSpanProcessor;
import org.jspecify.annotations.Nullable;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledIfEnvironmentVariable;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import reactor.core.publisher.Hooks;

import org.springframework.ai.chat.client.ChatClient;
import org.springframework.ai.chat.client.DefaultChatClientBuilder;
import org.springframework.ai.chat.client.advisor.ToolCallAdvisor;
import org.springframework.ai.chat.client.advisor.observation.AdvisorObservationContext;
import org.springframework.ai.chat.client.observation.ChatClientObservationContext;
import org.springframework.ai.chat.observation.ChatModelObservationContext;
import org.springframework.ai.model.tool.ToolCallingManager;
import org.springframework.ai.openai.OpenAiChatModel;
import org.springframework.ai.openai.OpenAiChatOptions;
import org.springframework.boot.SpringBootConfiguration;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.test.context.ActiveProfiles;

import static org.assertj.core.api.Assertions.assertThat;

@SpringBootTest(classes = OpenAiObservationOtelIT.Config.class)
@EnabledIfEnvironmentVariable(named = "OPENAI_API_KEY", matches = ".+")
@ActiveProfiles("logging-test")
class OpenAiObservationOtelIT {

	private static final Logger logger = LoggerFactory.getLogger(OpenAiObservationOtelIT.class);

	static final String CHAT_MODEL_STREAM_ADVISOR_NAME = "stream";

	// =========================================================================
	// Shared assertion helpers (must appear before inner types per InnerTypeLast)
	// =========================================================================

	private static void assertChatClientIsRoot(ObservationCapture capture) {
		var chatClientNode = capture.findChatClient();
		assertThat(chatClientNode).as("ChatClientObservationContext must be present").isPresent();
		assertThat(chatClientNode.get().parentCtx()).as("spring.ai.chat.client must be the root span (no parent)")
			.isNull();
	}

	// =========================================================================
	// Per-test helper methods
	// =========================================================================

	private ChatClient buildChatClient(ObservationRegistry registry) {
		var model = OpenAiChatModel.builder()
			.observationRegistry(registry)
			.options(OpenAiChatOptions.builder()
				.apiKey(System.getenv("OPENAI_API_KEY"))
				.model(OpenAiChatOptions.DEFAULT_CHAT_MODEL)
				.build())
			.build();

		ToolCallAdvisor.Builder<?> toolCallAdvisorBuilder = ToolCallAdvisor.builder()
			.toolCallingManager(ToolCallingManager.builder().observationRegistry(registry).build());

		return new DefaultChatClientBuilder(model, registry, null, null, toolCallAdvisorBuilder).build();
	}

	private void logCapture(ObservationCapture capture) {
		logger.info("=== Observation timeline ({} events) ===", capture.fullTimeline().size());
		capture.fullTimeline().forEach(event -> {
			String kind = switch (event.kind()) {
				case START -> "START      ";
				case SCOPE_OPEN -> "SCOPE_OPEN ";
				case SCOPE_CLOSE -> "SCOPE_CLOSE";
			};
			String ctxLabel = ctxLabel(event.ctx());
			String parentLabel = event.parentCtx() != null ? " parent=" + ctxLabel(event.parentCtx()) : "";
			String thread = " [" + event.thread() + "]";
			logger.info("  {} {}{}{}", kind, ctxLabel, parentLabel, thread);
		});
	}

	private static String ctxLabel(Observation.Context ctx) {
		String type = ctx.getClass().getSimpleName();
		if (ctx instanceof AdvisorObservationContext a) {
			return type + "(" + a.getAdvisorName() + ")";
		}
		return type;
	}

	private static String expectedSpanName(Observation.Context ctx) {
		if (ctx instanceof ChatClientObservationContext) {
			return "spring_ai chat_client";
		}
		if (ctx instanceof AdvisorObservationContext advisorCtx) {
			return advisorCtx.getAdvisorName();
		}
		if (ctx instanceof ChatModelObservationContext chatModelCtx) {
			return "chat " + chatModelCtx.getRequest().getOptions().getModel();
		}
		return ctx.getName();
	}

	@Test
	void streamObservationTreeEnabledContextPropagation() {
		try {
			Hooks.enableAutomaticContextPropagation();
			streamObservationTree();
		}
		finally {
			Hooks.disableAutomaticContextPropagation();
		}
	}

	@Test
	void streamObservationTreeDisabledContextPropagation() {
		streamObservationTree();
	}

	private void streamObservationTree() {

		var reg = TestObservationRegistry.create();
		var capture = new ObservationCapture();

		InMemorySpanExporter exporter = InMemorySpanExporter.create();
		SdkTracerProvider tracerProvider = SdkTracerProvider.builder()
			.addSpanProcessor(SimpleSpanProcessor.create(exporter))
			.build();

		OpenTelemetrySdk openTelemetry = OpenTelemetrySdk.builder().setTracerProvider(tracerProvider).build();

		Tracer otelApiTracer = openTelemetry.getTracer("test");

		OtelCurrentTraceContext otelCurrentTraceContext = new OtelCurrentTraceContext();

		OtelTracer tracer = new OtelTracer(otelApiTracer, otelCurrentTraceContext, event -> {
		}, new OtelBaggageManager(otelCurrentTraceContext, Collections.emptyList(), Collections.emptyList()));

		reg.observationConfig().observationHandler(new DefaultTracingObservationHandler(tracer));
		reg.observationConfig().observationHandler(capture.handler());

		String response = buildChatClient(reg).prompt()
			.user("Say `Hello`")
			.stream()
			.content()
			.collectList()
			.block()
			.stream()
			.collect(Collectors.joining());

		logger.info(response);

		logCapture(capture);

		assertChatClientIsRoot(capture);

		assertThat(capture.nodes.get(0).ctx()).as("ChatClientObservationContext must be the root span")
			.isInstanceOf(ChatClientObservationContext.class);
		assertThat(capture.nodes.get(0).parentCtx())
			.as("ChatClientObservationContext must be the root span (no parent)")
			.isNull();

		assertThat(capture.nodes.get(1).ctx()).as("Second observation must be the 'stream' advisor span")
			.isInstanceOf(AdvisorObservationContext.class)
			.satisfies(ctx -> assertThat(((AdvisorObservationContext) ctx).getAdvisorName())
				.isEqualTo(CHAT_MODEL_STREAM_ADVISOR_NAME));
		assertThat(capture.nodes.get(1).parentCtx()).as("'stream' advisor  parent must be ChatClientObservationContex")
			.isInstanceOf(ChatClientObservationContext.class);

		assertThat(capture.nodes.get(2).ctx()).as("Third observation must be the ChatModel span")
			.isInstanceOf(ChatModelObservationContext.class);
		assertThat(capture.nodes.get(2).parentCtx()).as("ChatModel span parent must be the 'stream' advisor")
			.isInstanceOf(AdvisorObservationContext.class)
			.satisfies(ctx -> assertThat(((AdvisorObservationContext) ctx).getAdvisorName())
				.isEqualTo(CHAT_MODEL_STREAM_ADVISOR_NAME));

		var chatModelNodes = capture.findAllByType(ChatModelObservationContext.class);
		assertThat(chatModelNodes).as("Expected at least 1 LLM calls").hasSizeGreaterThanOrEqualTo(1);

		List<SpanData> spans = exporter.getFinishedSpanItems();

		logger.info("--- Otel Spans ---");
		for (SpanData span : spans) {
			logger.info("Span: {} - context: {} - parent: {}", span.getName(), span.getSpanId(),
					span.getParentSpanId());
		}
		logger.info("------------------");

		assertThat(spans).as("Number of spans should match number of observations").hasSize(capture.nodes.size());

		List<String> obsTreeLinks = capture.nodes.stream().map(n -> {
			String parentName = n.parentCtx() != null ? expectedSpanName(n.parentCtx()) : "root";
			return parentName + " -> " + expectedSpanName(n.ctx());
		}).sorted().toList();

		Map<String, SpanData> spansById = spans.stream().collect(Collectors.toMap(s -> s.getSpanId(), s -> s));

		List<String> spanTreeLinks = spans.stream().map(s -> {
			String parentName = "root";
			String parentId = s.getParentSpanId();
			if (parentId != null && !parentId.isEmpty() && !parentId.equals("0000000000000000")) {
				SpanData parentSpan = spansById.get(parentId);
				parentName = parentSpan != null ? parentSpan.getName() : "unknown";
			}
			return parentName + " -> " + s.getName();
		}).sorted().toList();

		assertThat(spanTreeLinks).as("Span hierarchy should match observation hierarchy").isEqualTo(obsTreeLinks);
	}

	static final class ObservationCapture {

		final List<Node> nodes = new CopyOnWriteArrayList<>();

		private final List<TimelineEvent> timeline = new CopyOnWriteArrayList<>();

		ObservationHandler<Observation.Context> handler() {
			return new ObservationHandler<>() {
				@Override
				public void onStart(Observation.Context ctx) {
					Observation.Context parentCtx = null;
					var parentObs = ctx.getParentObservation();
					if (parentObs != null) {
						parentCtx = (Observation.Context) parentObs.getContextView();
					}
					nodes.add(new Node(ctx, parentCtx));
					timeline.add(new TimelineEvent(Kind.START, ctx, parentCtx, thread()));
				}

				@Override
				public void onScopeOpened(Observation.Context ctx) {
					timeline.add(new TimelineEvent(Kind.SCOPE_OPEN, ctx, null, thread()));
				}

				@Override
				public void onScopeClosed(Observation.Context ctx) {
					timeline.add(new TimelineEvent(Kind.SCOPE_CLOSE, ctx, null, thread()));
				}

				@Override
				public boolean supportsContext(Observation.Context ctx) {
					return true;
				}

				private String thread() {
					return Thread.currentThread().getName();
				}
			};
		}

		List<Node> all() {
			return Collections.unmodifiableList(this.nodes);
		}

		List<TimelineEvent> fullTimeline() {
			return Collections.unmodifiableList(this.timeline);
		}

		Optional<Node> findChatClient() {
			return this.nodes.stream().filter(n -> n.ctx() instanceof ChatClientObservationContext).findFirst();
		}

		Optional<Node> findAdvisor(String name) {
			return this.nodes.stream()
				.filter(n -> n.ctx() instanceof AdvisorObservationContext a && a.getAdvisorName().equals(name))
				.findFirst();
		}

		List<Node> findAllAdvisors(String name) {
			return this.nodes.stream()
				.filter(n -> n.ctx() instanceof AdvisorObservationContext a && a.getAdvisorName().equals(name))
				.toList();
		}

		List<Node> findAllByType(Class<? extends Observation.Context> type) {
			return this.nodes.stream().filter(n -> type.isInstance(n.ctx())).toList();
		}

		enum Kind {

			START, SCOPE_OPEN, SCOPE_CLOSE

		}

		record TimelineEvent(Kind kind, Observation.Context ctx, Observation.@Nullable Context parentCtx,
				String thread) {
		}

		record Node(Observation.Context ctx, Observation.@Nullable Context parentCtx) {
		}

	}

	@SpringBootConfiguration
	static class Config {

	}

}
