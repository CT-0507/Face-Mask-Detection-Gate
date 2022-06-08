const mongoose = require('mongoose');
const Schema = mongoose.Schema;

const logSchema = new Schema(
    {
        _id: { type: mongoose.Types.ObjectId, required: true },
        deviceId: { type: String, required: true, ref: 'Device' },
        ip: { type: String, required: true },
        name: { type: String, required: true },
        sensor: { type: String, required: true },
        value: { type: Number, required: true },
    },
    {
        timestamps: true,
    },
);

const Log = mongoose.model('Logs', logSchema);
module.exports = Log;
