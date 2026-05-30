class IntentRouter {

    constructor(
        groq,
        model
    ) {

        this.groq =
            groq;

        this.model =
            model;
    }

    async classify(
        query
    ) {

        const prompt = `

You are an intent classifier.

Possible intents:

- rag
- web
- notes
- planner
- pdf_metadata
- memory
- general

Rules:

web:
Needs current information from the internet.

rag:
Requires uploaded PDFs.

notes:
Generate notes from uploaded PDFs.

planner:
Generate study plan from uploaded PDFs.

pdf_metadata:
Questions about uploaded PDFs themselves.

memory:
Questions about conversation memory.

general:
Everything else.

Return ONLY valid JSON.

Example:

{
  "intent":"web",
  "confidence":0.95
}

Query:

${query}

`;

        try {

            const response =
                await this.groq.chat
                    .completions.create({

                        model:
                            this.model,

                        temperature: 0,

                        messages: [
                            {
                                role: 'user',
                                content:
                                    prompt
                            }
                        ]
                    });

            return JSON.parse(
                response.choices[0]
                    .message.content
            );

        } catch {

            return {

                intent:
                    'general',

                confidence:
                    0
            };
        }
    }
}

module.exports =
    IntentRouter;