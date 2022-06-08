const mongoose = require('mongoose');
const User = require('../../models/user.js');
var session;
class MainController {
    index(req, res, next) {
        session = req.session;
        if (session.userid) {
            console.log('OK');
            res.render('main/show');
        } else res.redirect('/');
    }
}

module.exports = new MainController();
