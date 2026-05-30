const mongoose =
    require('mongoose');

const SessionContextSchema =
    new mongoose.Schema({

        userId: {
            type: String,
            required: true,
            unique: true,
            index: true
        },

        currentDocument: {
            type: String,
            default: null
        },

        currentTopic: {
            type: String,
            default: null
        },

        currentEntity: {
            type: String,
            default: null
        },

        lastIntent: {
            type: String,
            default: null
        },

        updatedAt: {
            type: Date,
            default: Date.now
        }

    });

module.exports =
    mongoose.model(
        'SessionContext',
        SessionContextSchema
    );