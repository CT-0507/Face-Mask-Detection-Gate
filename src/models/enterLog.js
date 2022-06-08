const mongoose = require('mongoose');
const Schema = mongoose.Schema;

const enterLogSchema = new Schema(
    {
        _id: { type: mongoose.Types.ObjectId, required: true },
        attachTo: {
            type: mongoose.Types.ObjectId,
            ref: 'Devices',
        },
        ip: { type: String, required: true },
        name: { type: String, required: true },
        img: {
            data: Buffer,
            contentType: String,
        },
    },
    {
        timestamps: true,
    },
);

const EnterLogs = mongoose.model('EnterLogs', enterLogSchema);
module.exports = EnterLogs;
