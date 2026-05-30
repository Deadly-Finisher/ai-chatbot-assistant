class DocumentService {

    constructor(
        mongoPdfRepo
    ) {

        this.repo =
            mongoPdfRepo;
    }

    async getAllDocuments(
        userId
    ) {

        return await this.repo
            .getUserDocuments(
                userId
            );
    }

    async getLatestDocument(
        userId
    ) {

        const docs =
            await this.repo
                .getLatestUserDocuments(
                    userId,
                    1
                );

        return docs[0] || null;
    }

    async getDocumentCount(
        userId
    ) {

        const docs =
            await this.repo
                .getUserDocuments(
                    userId
                );

        return docs.length;
    }

    async getDocumentByFilename(
        userId,
        filename
    ) {

        return await this.repo
            .getDocumentByFilename(
                userId,
                filename
            );
    }

    async getDocumentNames(
        userId
    ) {

        const docs =
            await this.repo
                .getUserDocuments(
                    userId
                );

        return docs.map(
            d => d.filename
        );
    }
    async addDocument(
        document
    ) {

        return await this.repo
            .addDocument(
                document
            );
    }
}

module.exports =
    DocumentService;