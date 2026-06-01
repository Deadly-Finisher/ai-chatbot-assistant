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

Return ONLY the rewritten query.

If no rewrite is needed,
return the original query.
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

            return result
                .choices[0]
                .message.content
                .trim();

        } catch {

            return query;
        }
    }
}

module.exports =
    ContextAgent;