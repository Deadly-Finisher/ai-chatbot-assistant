require('dotenv').config();


const client =
    require('./qdrant');

async function test() {

    try {

        await client.upsert(
            'pdf_embeddings',
            {
                wait: true,

                points: [

                    {
                        id: 1,

                        vector:
                            Array(384).fill(0.1),

                        payload: {

                            text:
                                'Hello Qdrant',

                            filename:
                                'test.pdf'
                        }
                    }

                ]
            }
        );

        console.log(
            '✅ Test vector inserted'
        );
        const result =
            await client.search(
                'pdf_embeddings',
                {
                    vector:
                        Array(384).fill(0.1),

                    limit: 1
                }
            );

        console.log(
            'SEARCH RESULT:'
        );

        console.log(
            JSON.stringify(
                result,
                null,
                2
            )
        );

    } catch (err) {

        console.error(err);
    }
}

test();