const mongoose = require('mongoose');
const User = require('../../models/user.js');
const Log = require('../../models/log.js');
const DHT = require('../../models/dht.js');
const Soil = require('../../models/soil.js');
const BH = require('../../models/bh.js');
var session;
class LogController {
    index(req, res, next) {
        session = req.session;
        if (session.userid) {
            console.log('OK');
            //res.render('log/show');
            let perPage = 6; // số lượng sản phẩm xuất hiện trên 1 page
            let page = req.query.page || 1;
            if (page == 0) page = 1;
            //let page = 1 || 1;
            var title = 'Logs';
            var logsObject = [];
            Log.find() // find tất cả các data
                .skip(perPage * page - perPage) // Trong page đầu tiên sẽ bỏ qua giá trị là 0
                .limit(perPage)
                .exec((err, logs) => {
                    Log.countDocuments((err, count) => {
                        // đếm để tính có bao nhiêu trang
                        if (err) return next(err);
                        logs.forEach((log) => {
                            logsObject.push({
                                _id: log._id,
                                deviceId: log.deviceId,
                                ip: log.ip,
                                name: log.name,
                                sensor: log.sensor,
                                value: log.value,
                                createdAt: log.createdAt.toLocaleDateString(),
                            });
                        });
                        //console.log(logsObject);
                        res.render('log/show', {
                            title: title,
                            logsObject: logsObject,
                            current: page,
                            pages: Math.ceil(count / perPage),
                        }); // Trả về dữ liệu các sản phẩm theo định dạng như JSON, XML,...
                    });
                });
        } else res.redirect('/');
    }
    // [DELETE] /logs/:id
    destroy(req, res, next) {
        Log.deleteOne({ _id: req.params.id })
            .then(() => res.redirect('back'))
            .catch(next);
    }
    // [DELETE] /logs/delete-all
    destroyAll(req, res, next) {
        console.log('Im at delete all');
        BH.remove({}, function (err, result) {
            if (err) {
                console.log(err);
            } else {
                console.log(result);
            }
        });
        DHT.remove({}, function (err, result) {
            if (err) {
                console.log(err);
            } else {
                console.log(result);
            }
        });
        Soil.remove({}, function (err, result) {
            if (err) {
                console.log(err);
            } else {
                console.log(result);
            }
        });
        Log.remove({}, function (err, result) {
            if (err) {
                console.log(err);
                res.status(404).send({
                    message: 'Error: cannot connect to database',
                });
            } else {
                console.log(result);
                res.redirect('/logs/');
            }
        });
    }
    // show(req, res, next) {
    //     let perPage = 6; // số lượng sản phẩm xuất hiện trên 1 page
    //     let page = req.query.page || 1;
    //     //let page = 1 || 1;
    //     var title = 'Logs';
    //     var logsObject = [];
    //     Log.find() // find tất cả các data
    //         .skip(perPage * page - perPage) // Trong page đầu tiên sẽ bỏ qua giá trị là 0
    //         .limit(perPage)
    //         .exec((err, logs) => {
    //             Log.countDocuments((err, count) => {
    //                 // đếm để tính có bao nhiêu trang
    //                 if (err) return next(err);
    //                 logs.forEach((log) => {
    //                     logsObject.push({
    //                         _id: log._id,
    //                         deviceId: log.deviceId,
    //                         ip: log.ip,
    //                         name: log.name,
    //                         sensor: log.sensor,
    //                         value: log.value,
    //                         createdAt: log.createdAt.toLocaleDateString(),
    //                     });
    //                 });
    //                 console.log(logsObject);
    //                 res.render('log/show', {
    //                     title: title,
    //                     logsObject: logsObject,
    //                     current: page,
    //                     pages: Math.ceil(count / perPage),
    //                 }); // Trả về dữ liệu các sản phẩm theo định dạng như JSON, XML,...
    //             });
    //         });
    // }
    filter(req, res, next) {
        session = req.session;
        if (session.userid) {
            let perPage = 6; // số lượng sản phẩm xuất hiện trên 1 page
            let page = req.query.page || 1;
            if (page == 0) page = 1;
            //let page = 1 || 1;
            var title = 'Logs';
            var logsObject = [];
            console.log(req.query.q);
            Log.find({ $or: [{ name: req.query.q }, { ip: req.query.q }] }) // find tất cả các data
                .skip(perPage * page - perPage) // Trong page đầu tiên sẽ bỏ qua giá trị là 0
                .limit(perPage)
                .exec((err, logs) => {
                    Log.countDocuments(
                        { $or: [{ name: req.query.q }, { ip: req.query.q }] },
                        (err, count) => {
                            // đếm để tính có bao nhiêu trang
                            if (err) return next(err);
                            logs.forEach((log) => {
                                logsObject.push({
                                    _id: log._id,
                                    deviceId: log.deviceId,
                                    ip: log.ip,
                                    name: log.name,
                                    sensor: log.sensor,
                                    value: log.value,
                                    createdAt:
                                        log.createdAt.toLocaleDateString(),
                                });
                            });
                            //console.log(logsObject);
                            res.render('log/filter', {
                                title: title,
                                logsObject: logsObject,
                                current: page,
                                pages: Math.ceil(count / perPage),
                                filterValue: req.query.q,
                            }); // Trả về dữ liệu các sản phẩm theo định dạng như JSON, XML,...
                        },
                    );
                });
        } else {
            res.redirect('/');
        }
    }
}

module.exports = new LogController();
