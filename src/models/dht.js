const mongoose = require('mongoose');
const Schema = mongoose.Schema;

const dhtSchema = new Schema(
    {
        _id: { type: mongoose.Types.ObjectId, required: true },
        name: { type: String, required: true },
        temp: { type: Number, required: true },
        humi: { type: Number, required: true },
        // change date
        attachTo: {
            type: mongoose.Types.ObjectId,
            required: true,
            ref: 'Devices',
        },
        createdAt: {
            type: Date,
            default: Date.now,
        },
        updatedAt: {
            type: Date,
            default: Date.now,
        },
    },
    // {
    //     timestamps: true,
    // },
);

const DHT = mongoose.model('dht', dhtSchema);
module.exports = DHT;
