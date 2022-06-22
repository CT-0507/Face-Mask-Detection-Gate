const mongoose = require('mongoose');
let host = '172.31.250.62';
const EnterLogs = require('../../models/enterLog.js');
var session;
class ChartsController {
    // GET /
    async index(req, res, next) {
        session = req.session;
        if (session.userid) {
            var title = 'Charts';
            var withMask = 0;
            var noMask = 0;
            const data = [];
            var ChartsData;
            try {
                var today = new Date();
                var date = today.getMonth() + '/' + today.getDate() + '/' + today.getFullYear();
                console.log(date)
                try {
                    var Logs = await EnterLogs.find();
                    Logs.forEach((result) => {
                        if(result.withMask) {
                            withMask++;
                        } else noMask++;
                    })
                    ChartsData = {
                        withMask: withMask,
                        noMask: noMask,
                    }
                    console.log(ChartsData);
                    var daysData = {
                        firstday: {
                            withMask: 0,
                            noMask: 0,
                        },
                        secondday: {
                            withMask: 0,
                            noMask: 0,
                        },
                        thirdday: {
                            withMask: 0,
                            noMask: 0,
                        },
                        fourthday: {
                            withMask: 0,
                            noMask: 0,
                        },
                    }
                    var firstday = await EnterLogs.find({
                        createdAt: {
                            $gte: new Date(Date.now() - 60 * 60 * 24 * 1000)
                        }
                    });
                    firstday.forEach((result) => {
                        if(result.withMask) {
                            daysData.firstday.withMask++;
                        } else daysData.firstday.noMask++;
                    });
                    var secondday = await EnterLogs.find({
                        createdAt: {
                            $gte: new Date(new Date() - 2* 60 * 60 * 24 * 1000),
                            $lte: new Date(new Date() - 60 * 60 * 24 * 1000)
                        }
                    });
                    secondday.forEach((result) => {
                        if(result.withMask) {
                            daysData.secondday.withMask++;
                        } else daysData.secondday.noMask++;
                    });
                    var thirdday = await EnterLogs.find({
                        createdAt: {
                            $gte: new Date(new Date() - 3* 60 * 60 * 24 * 1000),
                            $lte: new Date(new Date() - 2*60 * 60 * 24 * 1000)
                        }
                    });
                    thirdday.forEach((result) => {
                        if(result.withMask) {
                            daysData.thirdday.withMask++;
                        } else daysData.thirdday.noMask++;
                    });
                    var fourthday = await EnterLogs.find({
                        createdAt: {
                            $gte: new Date(new Date() - 4* 60 * 60 * 24 * 1000),
                            $lte: new Date(new Date() - 3*60 * 60 * 24 * 1000)
                        }
                    });
                    fourthday.forEach((result) => {
                        if(result.withMask) {
                            daysData.fourthday.withMask++;
                        } else daysData.fourthday.noMask++;
                    });
                    console.log(daysData);
                }
                catch(err){
                    console.log(err);
                }
                res.render('charts/show', { ChartsData: ChartsData, title: title, host: host, daysData: daysData });
            }
            catch (err) {
                res.status(404).send({
                    message: 'error',
                });
            }
        }
        else {
            res.redirect('/');
        }
    }
    async getdata(req, res, next) {
        try {
            const dht = await DHT.findOne().sort({ createdAt: -1 });
            const bh = await BH.findOne().sort({ createdAt: -1 });
            const sl = await SL.findOne().sort({ createdAt: -1 });
            let data = {
                temp: dht.temp,
                humi: dht.humi,
                lux: bh.lux,
                soilHumi: sl.soilHumi,
                tempCreatedAt: dht.createdAt,
                humiCreatedAt: dht.createdAt,
                soilCreatedAt: sl.createdAt,
                luxCreatedAt: bh.createdAt,
            };
            console.log(data);
            res.json({ data: data });
        } catch (err) {
            res.status(404).send({
                message: 'err',
            });
        }
    }
}

module.exports = new ChartsController();
