class ContextAgent {

    constructor(groq, model) {

        this.groq = groq;
        this.model = model;
    }

    async resolve(
        query,
        history,
        currentDocument = null
    ) {

        const prompt = `
You are a context resolution agent.

Recent Conversation:

${history}

Current Active Document:

${currentDocument || "None"}

Current Query:

${query}

If the query contains references like:

- it
- its
- they
- them
- this
- that

If the query refers to previously mentioned documents,
PDFs, papers, files, or collections using words such as:

- them
- those
- all of them
- show them
- list them

rewrite the query into a fully explicit document request.

Examples:

How many PDFs do I have?

show them

becomes

show all uploaded pdfs

List my uploaded papers

name them

becomes

list all uploaded pdfs

rewrite the query so it becomes fully explicit.

Examples:

latest fc barcelona squad
→ tell me its full squad
becomes
→ tell me the full FC Barcelona squad

If a Current Active Document exists and the query refers to:

- it
- its
- this pdf
- the pdf
- this document
- the document

then rewrite the query using the document name.

Examples:

Current Active Document:
70mb.pdf

summarize it
becomes
summarize 70mb.pdf

give me notes from it
becomes
give me notes from 70mb.pdf

what is this pdf about
becomes
what is 70mb.pdf about


If a Current Active Document exists and the query contains:

- it
- its
- this pdf
- the pdf
- this document
- the document

then rewrite the query using the Current Active Document.

Examples:

Current Active Document:
BLIP-2.pdf

summarize it

summarize BLIP-2.pdf

what is this pdf about

what is BLIP-2.pdf about

give me notes from it

give me notes from BLIP-2.pdf

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