class PDFAgent {

    constructor(groq, model) {
        this.groq = groq;
        this.model = model;
    }

    async classify(query) {

        const prompt = `
  You are a PDF intent classifier.
  
  Query:
  ${query}
  
  Return ONLY valid JSON.
  
  Possible intents:
  
  paper_summary
  paper_contributions
  paper_methodology
  paper_results
  paper_limitations
  paper_datasets
  paper_future_work
  paper_general_qa
  
  Example:
  
  {
    "intent":"paper_summary"
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
                intent:
                    "paper_general_qa"
            };
        }
    }
}

module.exports =
    PDFAgent;