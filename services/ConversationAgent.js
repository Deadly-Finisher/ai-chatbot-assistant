class ConversationAgent {

    constructor(groq, model) {

        this.groq = groq;
        this.model = model;
    }

    async classify(query) {

        const prompt = `
You are a conversation intent classifier.

User Query:
${query}

Return ONLY JSON.

Possible intents:

smalltalk
general

Examples:

hi

{
  "intent":"smalltalk",
  "response":"Hello 👋"
}

hello

{
  "intent":"smalltalk",
  "response":"Hello 👋"
}

how are you

{
  "intent":"smalltalk",
  "response":"I'm functioning perfectly 🚀"
}

thanks

{
  "intent":"smalltalk",
  "response":"You're welcome 🚀"
}

what is machine learning

{
  "intent":"general"
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

            const cleaned =
                raw
                    .replace(/```json/g, '')
                    .replace(/```/g, '')
                    .trim();

            return JSON.parse(cleaned);

        } catch {

            return {
                intent: "general"
            };
        }
    }
}

module.exports =
    ConversationAgent;