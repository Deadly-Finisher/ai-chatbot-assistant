const client =
    require('./qdrant');

async function initQdrant() {

    try {

        const collections =
            await client.getCollections();

        const exists =
            collections.collections.some(
                c =>
                    c.name ===
                    'pdf_embeddings'
            );

        if (!exists) {

            await client.createCollection(
                'pdf_embeddings',
                {
                    vectors: {
                        size: 384,
                        distance: 'Cosine'
                    }
                }
            );

            console.log(
                '✅ Qdrant collection created'
            );

        } else {

            console.log(
                '✅ Qdrant collection exists'
            );
        }

        // userId index
        try {

            await client.createPayloadIndex(
                'pdf_embeddings',
                {
                    field_name: 'userId',
                    field_schema: 'keyword'
                }
            );

            console.log(
                '✅ userId index created'
            );

        } catch (err) {

            console.log(
                'ℹ️ userId index already exists'
            );
        }

        // filename index
        try {

            await client.createPayloadIndex(
                'pdf_embeddings',
                {
                    field_name: 'filename',
                    field_schema: 'keyword'
                }
            );

            console.log(
                '✅ filename index created'
            );

        } catch (err) {

            console.log(
                'ℹ️ filename index already exists'
            );
        }

    } catch (err) {

        console.error(
            'Qdrant Init Error:',
            err.message
        );
    }
}

module.exports =
    initQdrant;

if (require.main === module) {

    initQdrant()
        .then(() => {

            console.log(
                '✅ Qdrant initialization complete'
            );

            process.exit(0);
        })
        .catch(err => {

            console.error(err);

            process.exit(1);
        });
}