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