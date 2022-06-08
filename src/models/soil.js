const mongoose = require('mongoose');
const Schema = mongoose.Schema;

const soilSchema = new Schema(
    {
        _id: { type: mongoose.Types.ObjectId, required: true },
        name: { type: String, required: true },
        soilHumi: { type: Number, required: true },
        // change date
        attachTo: {
            type: mongoose.Types.ObjectId,
            required: true,
            ref: 'Devices',
        },
    },
    {
        timestamps: true,
    },
);

const soil = new mongoose.model('soil', soilSchema);
module.exports = soil;
