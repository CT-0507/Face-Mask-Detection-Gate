const mongoose = require('mongoose');
const Schema = mongoose.Schema;

const deviceSchema = new Schema(
    {
        _id: { type: mongoose.Types.ObjectId, require: true },
        name: { type: String, require: true },
        imgUrl: { type: String, require: true },
    },
    {
        timestamps: true,
    },
);
const Device = mongoose.model('Devices', deviceSchema);
module.exports = Device;
