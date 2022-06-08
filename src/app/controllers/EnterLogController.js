const mongoose = require('mongoose');
const User = require('../../models/user.js');
const EnterLogs = require('../../models/enterLog.js');
const fs = require('fs');
const path = require('path');
const {
    mongooseToObject,
    multipleMongooseToObject,
} = require('../../util/mongoose');
var session;
class EnterLogController {
    // [DELETE] delete enterlog
    destroy(req, res, next) {
        EnterLogs.deleteOne({ _id: req.params.id })
            .then(() => res.redirect('back'))
            .catch(next);
    }
    // [DELETE] enterlog/delete-all
    destroyAll(req, res, next) {
        console.log('Im at delete all');
        EnterLogs.remove({}, function (err, result) {
            if (err) {
                console.log(err);
                res.status(404).send({
                    message: 'Error: cannot connect to database',
                });
            } else {
                console.log(result);
                res.redirect('/enterlog/');
            }
        });
    }

    detail(req, res, next) {
        session = req.session;
        const title = 'Enter Log/Detail';
        if (session.userid) {
            EnterLogs.findOne({ _id: req.params.id }, (err, item) => {
                if (err) {
                    console.log(err);
                    res.status(500).send('An error occurred', err);
                } else {
                    var newItem = {
                        _id: item._id,
                        attachTo: item.attachTo,
                        ip: item.ip,
                        name: item.name,
                        img: item.img,
                        createdAt: item.createdAt.toLocaleDateString(),
                    };
                    //console.log(newItem);
                    res.render('enterlog/detail', {
                        items: newItem,
                        title: title,
                    });
                }
            });
        } else {
            res.redirect('/');
        }
    }
    //[GEt] show
    index(req, res, next) {
        session = req.session;
        if (session.userid) {
            var itemsObject = [];
            let perPage = 6; // số lượng sản phẩm xuất hiện trên 1 page
            let page = req.query.page || 1;
            if (page == 0) page = 1;
            //let page = 1 || 1;
            var title = 'Enter Logs';
            EnterLogs.find() // find tất cả các data
                .skip(perPage * page - perPage) // Trong page đầu tiên sẽ bỏ qua giá trị là 0
                .limit(perPage)
                .exec((err, items) => {
                    EnterLogs.countDocuments((err, count) => {
                        // đếm để tính có bao nhiêu trang
                        if (err) return next(err);
                        items.forEach((item) => {
                            itemsObject.push({
                                _id: item._id,
                                attachTo: item.attachTo,
                                ip: item.ip,
                                name: item.name,
                                //img: item.img,
                                createdAt: item.createdAt.toLocaleDateString(),
                            });
                        });
                        //console.log(itemsObject);
                        res.render('enterlog/show', {
                            title: title,
                            itemsObject: itemsObject,
                            current: page,
                            pages: Math.ceil(count / perPage),
                        }); // Trả về dữ liệu các sản phẩm theo định dạng như JSON, XML,...
                    });
                });
        } else {
            console.log('here2');
            res.redirect('/');
        }
    }
    // [POST] upload image /enterlog/upload
    upload(req, res, next) {
        console.log(path.join(__dirname, '..', '..', 'uploads'));
        console.log(req.body);
        var obj = {
            _id: new mongoose.Types.ObjectId(),
            name: req.body.name,
            attachTo: req.body.attachTo,
            ip: req.body.ip || '172.0.0.0',
            img: {
                data: fs.readFileSync(
                    path.join(
                        __dirname,
                        '..',
                        '..',
                        '/uploads/' + req.file.filename,
                    ),
                ),
                contentType: 'image/png',
            },
        };
        console.log("I'm here");
        EnterLogs.create(obj, (err, item) => {
            if (err) {
                console.log(err);
            } else {
                // item.save();
                res.redirect('/enterlog');
            }
        });
    }
    filter(req, res, next) {
        session = req.session;
        if (session.userid) {
            let perPage = 6; // số lượng sản phẩm xuất hiện trên 1 page
            let page = req.query.page || 1;
            if (page == 0) page = 1;
            //let page = 1 || 1;
            var title = 'Enter Logs';
            var itemsObject = [];
            EnterLogs.find({
                $or: [{ name: req.query.q }, { ip: req.query.q }],
            }) // find tất cả các data
                .skip(perPage * page - perPage) // Trong page đầu tiên sẽ bỏ qua giá trị là 0
                .limit(perPage)
                .exec((err, items) => {
                    EnterLogs.countDocuments(
                        { $or: [{ name: req.query.q }, { ip: req.query.q }] },
                        (err, count) => {
                            // đếm để tính có bao nhiêu trang
                            if (err) return next(err);
                            items.forEach((item) => {
                                itemsObject.push({
                                    _id: item._id,
                                    attachTo: item.attachTo,
                                    ip: item.ip,
                                    name: item.name,
                                    createdAt:
                                        item.createdAt.toLocaleDateString(),
                                });
                            });
                            res.render('log/filter', {
                                title: title,
                                itemsObject: itemsObject,
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

module.exports = new EnterLogController();
