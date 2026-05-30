require('dotenv').config();

const {
    QdrantClient
} = require(
    '@qdrant/js-client-rest'
);

const client =
    new QdrantClient({

        url:
            process.env.QDRANT_URL,

        apiKey:
            process.env.QDRANT_API_KEY,

        checkCompatibility: false
    });

module.exports = client;