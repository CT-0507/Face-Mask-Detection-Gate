const mongoose = require('mongoose');
const Device = require('../../models/devices.js');
const DHT = require('../../models/dht.js');
const Soil = require('../../models/soil.js');
const BH = require('../../models/bh.js');
const Log = require('../../models/log.js');
const _io = require('../../index').io;

class UpdateController {
    // update/device
    async addDevice(req, res) {
        const data = new Device({
            _id: new mongoose.Types.ObjectId(),
            name: req.body.name,
            imgUrl: req.body.imgUrl,
        });

        data.save()
            .then((result) => {
                console.log(result);
                res.status(200).send(result._id.toString());
            })
            .catch((err) => {
                console.log('Fail to update to database');
                console.log(err);
                res.status(400).send('Fail to update to db.');
            });
    }
    // update/dht
    async updateDht(req, res) {
        const data = new DHT({
            _id: new mongoose.Types.ObjectId(),
            name: req.body.name,
            temp: req.body.temp,
            humi: req.body.humi,
            attachTo: req.body.attachTo,
        });
        _io.emit('sendDht', data); // for charts.hbs
        _io.emit('sendDhtMain', data) // for main.hbs

        data.save()
            .then((result) => {
                console.log('Update dht data to db.');
            })
            .catch((err) => {
                console.log(err);
                res.status(400).send(`Fail to update`);
                return;
            });

        // Lấy dữ liệu từ database
        const devicesInfo = await Device.findOne({ _id: req.body.attachTo });
        var deviceName = devicesInfo.name;
        const log1 = new Log({
            _id: new mongoose.Types.ObjectId(),
            deviceId: req.body.attachTo,
            ip: req.body.ip, // Hmmmmmmmmmmm
            name: deviceName,
            sensor: 'Nhiệt độ',
            value: parseInt(req.body.temp),
        });

        log1.save()
            .then(() => {
                console.log(log1);
                //res.send(log)
                console.log('Update temperature log');
            })
            .catch((err) => {
                // res.status(404).send('Failed to create');
                console.log(err);
                return;
            });
        const log2 = new Log({
            _id: new mongoose.Types.ObjectId(),
            deviceId: req.body.attachTo,
            ip: req.body.ip,
            name: deviceName,
            sensor: 'Độ ẩm',
            value: parseInt(req.body.humi),
        });
        log2.save()
            .then(() => {
                console.log('Update humidity log');
                res.status(200).send('Ok');
            })
            .catch((err) => {
                console.log(err);
                res.send('Failed to create');
            });
    }

    // update/bh
    async updateBh(req, res, next) {
        const data = new BH({
            _id: new mongoose.Types.ObjectId(),
            name: req.body.name,
            lux: req.body.lux,
            attachTo: req.body.attachTo,
        });

        _io.emit('sendBh', data)
        _io.emit('sendBhMain', data)
        data.save()
            .then((result) => {
                console.log(result);
                console.log(result._id.toString());

                // res.status(200).send(result._id.toString())
            })
            .catch((err) => {
                console.log(err);
                //res.status(400).send(`Fail to update`)
            });
        const devicesInfo = await Device.findOne({ _id: req.body.attachTo });
        var deviceName = devicesInfo.name;
        const log = new Log({
            _id: new mongoose.Types.ObjectId(),
            deviceId: req.body.attachTo,
            ip: req.body.ip,
            name: deviceName,
            sensor: ' Độ sáng',
            value: req.body.lux,
        });
        log.save()
            .then(() => {
                res.send(log);
            })
            .catch(() => {
                res.send('Failed to create');
            });
    }

    // update/soil
    async updateSoil(req, res) {
        const data = new Soil({
            _id: new mongoose.Types.ObjectId(),
            name: req.body.name,
            soilHumi: req.body.soilHumi,
            attachTo: req.body.attachTo,
        });

        _io.emit('sendSoilMain', data)

        data.save()
            .then((result) => {
                console.log(result);
                console.log(result._id.toString());

                // res.status(200).send(result._id.toString())
            })
            .catch((err) => {
                console.log(err);
                //res.status(400).send(`Fail to update`)
            });
        const devicesInfo = await Device.findOne({ _id: req.body.attachTo });
        var deviceName = devicesInfo.name;
        const log = new Log({
            _id: new mongoose.Types.ObjectId(),
            deviceId: req.body.attachTo,
            ip: req.body.ip,
            name: deviceName,
            sensor: 'Độ ẩm đất',
            value: req.body.soilHumi,
        });
        log.save()
            .then(() => {
                res.status(200).send(log);
            })
            .catch(() => {
                res.send('Failed to create');
            });
    }

    loging(req, res, next) {
        const log = new Log({
            _id: new mongoose.Types.ObjectId(),
            deviceId: req.body.attachTo,
            ip: req.body.ip,
            name: req.body.name,
            sensor: req.body.sensor,
            value: req.body.value,
        });
        log.save()
            .then(() => {
                res.send(log);
            })
            .catch(() => {
                res.send('Failed to create');
            });
    }
}

module.exports = new UpdateController();
