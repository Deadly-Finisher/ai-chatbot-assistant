const SessionContext =
    require('../models/SessionContext');

class DocumentContextService {

    async updateContext(
        userId,
        updates
    ) {

        return await SessionContext
            .findOneAndUpdate(

                { userId },

                {
                    ...updates,

                    updatedAt:
                        new Date()
                },

                {
                    upsert: true,
                    returnDocument: 'after'
                }
            );
    }

    async setCurrentDocument(
        userId,
        filename
    ) {

        return await this.updateContext(
            userId,
            {
                currentDocument:
                    filename,

                currentEntity:
                    filename
            }
        );
    }

    async getCurrentDocument(
        userId
    ) {

        const context =
            await SessionContext
                .findOne({
                    userId
                });

        return context
            ?.currentDocument || null;
    }

    async getContext(
        userId
    ) {

        return await SessionContext
            .findOne({
                userId
            });
    }
}

module.exports =
    new DocumentContextService();