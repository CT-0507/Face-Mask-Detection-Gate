const mongoose = require('mongoose')
const Schema = mongoose.Schema

const person = new Schema({
    _id: {type: mongoose.Types.ObjectId, required: true},
    withMask: {type: Boolean, required: true},
    attachTo:{type: mongoose.Types.ObjectId, required: true, ref: 'Devices'},
    img:{data: Buffer, contentType: String},
    ip:{type: String, required: true},
    createdAt: {
        type: Date,
        default: Date.now,
    },
    updatedAt: {
        type: Date,
        default: Date.now,
    },
})

const Person = mongoose.model("Persons", person)
module.exports = Person