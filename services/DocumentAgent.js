class DocumentAgent {

    constructor(groq, model) {

        this.groq = groq;
        this.model = model;
    }

    async classify(query) {

        const prompt = `
  You are a document intent classifier.
  
  User Query:
  ${query}

    DO NOT use markdown
    DO NOT wrap JSON in code blocks.
    DO NOT explain.
    Return exactly one JSON object.
  
  Return ONLY valid JSON.
  
  Possible intents:
  
  Possible intents:

latest_document
current_document
list_documents
document_count
document_query
unknown
  
  Examples:

  summarize the vit paper

{
  "intent":"document_query"
}

what are the contributions of medclip

{
  "intent":"document_query"
}

give me the summary of this paper

{
  "intent":"document_query"
}

  how many pdfs have i uploaded

{
  "intent":"document_count"
}
what is the current document

{
  "intent":"current_document"
}

which paper is active

{
  "intent":"current_document"
}
show all pdfs

{
  "intent":"list_documents"
}

list uploaded papers

{
  "intent":"list_documents"
}

  what is the latest pdf
  
  {
    "intent":"latest_document"
  }
  
  show all uploaded pdfs
  
  {
    "intent":"list_documents"
  }
  
  how many pdfs do i have
  
  {
    "intent":"document_count"
  }
  
  what is the current document
  
  {
    "intent":"current_document"
  }
  
  If unsure:
  
  {
    "intent":"unknown"
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
                '📄 DocumentAgent RAW:',
                raw
            );

            const cleaned =
                raw
                    .replace(/```json/g, '')
                    .replace(/```/g, '')
                    .trim();

            console.log(
                '📄 DocumentAgent CLEANED:',
                cleaned
                    );

            try {

                return JSON.parse(cleaned);

            } catch (parseErr) {

                console.error(
                    '❌ JSON Parse Error:',
                    parseErr
                );

                console.error(
                    '❌ Raw Response:',
                    cleaned
                );

                return {
                    intent: "unknown"
                };
                    }

        } catch (err) {

            console.error(
                '❌ DocumentAgent Error:',
                err
            );

            return {
                intent: "unknown"
            };
        }
    }
}

module.exports =
    DocumentAgent;