const Document =
    require('../models/Document');

class MongoDocumentRepository {

    async addDocument(doc) {

        return await Document.create({
            userId: doc.userId,
            filename: doc.filename,
            summary: doc.summary || '',
            path: doc.path,
            uploadedAt:
                doc.uploadedAt ||
                new Date()
        });
    }

    async getAll() {

        return await Document.find()
            .lean();
    }

    async getUserDocuments(
        userId
    ) {

        return await Document.find({
            userId
        }).lean();
    }
    async getLatestUserDocuments(
        userId,
        limit = 10
    ) {

        return await Document.find({
            userId
        })
            .sort({
                uploadedAt: -1
            })
            .limit(limit)
            .lean();
    }
    async getDocumentByFilename(
        userId,
        filename
    ) {

        return await Document.findOne({

            userId,

            filename

        }).lean();
    }
}



module.exports =
    MongoDocumentRepository;