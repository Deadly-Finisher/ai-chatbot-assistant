class ContextAgent {

    constructor(groq, model) {

        this.groq = groq;
        this.model = model;
    }

    async resolve(
        query,
        history
    ) {

        const prompt = `
You are a context resolution agent.

Recent Conversation:

${history}

Current Query:

${query}

If the query contains references like:

- it
- its
- they
- them
- this
- that

rewrite the query so it becomes fully explicit.

Examples:

latest fc barcelona squad
→ tell me its full squad
becomes
→ tell me the full FC Barcelona squad

Return ONLY ONE LINE.

Rules:
- Do not explain.
- Do not justify.
- Do not use arrows (→).
- Do not use examples.
- Do not include phrases like "no rewrite needed".
- Output must contain only the final rewritten query.

If no rewrite is needed, return the original query exactly.
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

            const resolved =
                result
                    .choices[0]
                    .message.content
                    .trim();

            const lines =
                resolved
                    .split('\n')
                    .map(line => line.trim())
                    .filter(Boolean);

            return lines[lines.length - 1];

        } catch {

            return query;
        }
    }
}

module.exports =
    ContextAgent;