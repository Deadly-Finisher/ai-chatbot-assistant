class PlannerAgent {

    constructor(groq, model) {

        this.groq = groq;
        this.model = model;
    }

    async plan(userMessage, state = {}) {

        const prompt = `
You are an AI planning agent.

Available tools:

WEB_SEARCH
PDF_SEARCH
MEMORY
TASKS

IMPORTANT RULES:

If the user asks about a document,
paper,
PDF,
research paper,
uploaded file,
or a currently active document,

ALWAYS prefer PDF_SEARCH.

Do NOT choose WEB_SEARCH if the
information is available in uploaded PDFs.

WEB_SEARCH should only be used when
the answer is not available in uploaded documents.

Current State:
${JSON.stringify(state, null, 2)}

User Query:
${userMessage}

Decide:

1. Which tools are needed.
2. In what order.
3. Whether multiple tools are required.

Return ONLY JSON.

Example:

{
  "tools":["WEB_SEARCH"],
  "reason":"Current information required"
}
`;

        try {

            const result =
                await this.groq.chat.completions.create({

                    model: this.model,

                    temperature: 0,

                    messages: [
                        {
                            role: "user",
                            content: prompt
                        }
                    ]
                });

            return JSON.parse(
                result.choices[0]
                    .message.content
            );

        } catch {

            return {
                tools: ["GENERAL"],
                reason: "Fallback"
            };
        }
    }
}

module.exports = PlannerAgent;