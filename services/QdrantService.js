const client =
    require('../qdrant');

class QdrantService {

    async storeChunk(
        point
    ) {

        await client.upsert(
            'pdf_embeddings',
            {
                wait: true,
                points: [point]
            }
        );
    }

    async search(
        vector,
        limit = 5,
        filename = null
    ) {

        const searchOptions = {

            vector,

            limit
        };

        if (filename) {

            searchOptions.filter = {

                must: [

                    {
                        key: 'filename',

                        match: {
                            value: filename
                        }
                    }
                ]
            };
        }

        return await client.search(
            'pdf_embeddings',
            searchOptions
        );
      }
}

module.exports =
    new QdrantService();