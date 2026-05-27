const fs = require('fs-extra');
const path = require('path');

class PDFRepository {

    constructor() {

        this.file =
            path.join(
                __dirname,
                '../data/pdfs.json'
            );

        this.data = {
            documents: []
        };

        this.load();
    }

    load() {

        if (
            fs.existsSync(this.file)
        ) {

            this.data =
                fs.readJsonSync(this.file);
        }
    }

    save() {

        fs.writeJsonSync(
            this.file,
            this.data,
            { spaces: 2 }
        );
    }

    addDocument(doc) {

        this.data.documents.push(doc);

        this.save();

        return doc;
    }

    getAll() {

        return this.data.documents;
    }

    getUserDocuments(userId) {

        return this.data.documents.filter(
            d => d.userId === userId
        );
    }
}

module.exports =
    PDFRepository;