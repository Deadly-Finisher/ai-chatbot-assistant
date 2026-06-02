class DocumentSelectionAgent {

    constructor(groq, model) {

        this.groq = groq;
        this.model = model;
    }

    async select(
        query,
        documents,
        currentDocument
    ) {

        const prompt = `
You are a document selection agent.

User Query:

${query}

Current Active Document:

${currentDocument || "None"}

Available Documents:

${documents.join('\n')}

Choose the SINGLE best document.

Rules:

- Return ONLY JSON.
- No markdown.
- No explanation.

Format:

{
  "selectedDocument":"filename.pdf"
}

If no document matches:

{
  "selectedDocument": null
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

            const raw =
                result.choices[0]
                    .message.content
                    .trim();

            console.log(
                '📄 DocumentSelection RAW:',
                raw
            );

            const cleaned =
                raw
                    .replace(/```json/g, '')
                    .replace(/```/g, '')
                    .trim();

            return JSON.parse(
                cleaned
            );

        } catch (err) {

            console.error(
                '❌ DocumentSelection Error:',
                err
            );

            return {
                selectedDocument: null
            };
        }
    }
}

module.exports =
    DocumentSelectionAgent;