const express = require('express');
const morgan = require('morgan');
const handlebars = require('express-handlebars');
const sass = require('node-sass');
const http = require('http');
const app = express();
const server = http.createServer(app);
const io = require('socket.io')(server);
exports.io = io;
const path = require('path');
const route = require('./routes');
const mongoose = require('mongoose');
const methodOverride = require('method-override');
const bodyParser = require('body-parser');
require('dotenv').config();
const fs = require('fs');
const multer = require('multer');
const host = '172.31.250.62';
async function connect() {
    try {
        await mongoose.connect(process.env.MONGO_URL, {
            // useNewUrlParser: true,
            // useUnifiedTopology: true
            //useCreateIndex: true,
        });
        console.log('Connect successfully');
    } catch (error) {
        console.log('Failed to connect');
    }
}

io.on('connect', (socket) => {
    console.log('User conenct to socket io');
});

app.use(function (req, res, next) {
    res.header('Access-Control-Allow-Origin', '*');
    res.header(
        'Access-Control-Allow-Headers',
        'Origin, X-Requested-With, Content-Type, Accept',
    );
    next();
});

const cookieParser = require('cookie-parser');
const sessions = require('express-session');
// session
// creating 24 hours from milliseconds
const oneDay = 1000 * 60 * 60 * 24;
app.use(
    sessions({
        secret: 'thisismysecrctekeyfhrgfgrfrty84fwir767',
        saveUninitialized: true,
        cookie: { maxAge: oneDay },
        resave: false,
    }),
);
app.use(function (req, res, next) {
    res.locals.session = req.session;
    next();
});
var session;
// cookie parser middleware
app.use(cookieParser());
// method override
app.use(methodOverride('_method'));
connect();
// use urlencoded
app.use(
    express.urlencoded({
        extended: true,
    }),
);
// JSON
app.use(express.json());
// public file can be view via url
app.use(express.static(path.join(__dirname, 'public')));
app.use(morgan('combined'));

//Template engine
// Using handlebars to view
app.engine(
    'hbs',
    handlebars.engine({
        extname: '.hbs',
        helpers: {
            checkCurrent: (a) => {
                if (a == 1) return true;
            },
            sum: function (a, b) {
                var c = parseInt(a) + parseInt(b);
                return c;
            },
            toString64: (a) => {
                var base = Buffer.from(a);
                var conversion = base.toString('base64');
                return conversion;
            },
        },
    }),
);

app.set('view engine', 'hbs');
app.set('views', path.join(__dirname, 'resources', 'views'));

// connect to route file
route(app);
// setup multer to upload image

const storage = multer.diskStorage({
    destination: (req, file, cb) => {
        cb(null, 'uploads');
    },
    filename: (req, file, cb) => {
        cb(null, file.fieldname + '-' + Date.now());
    },
});

const upload = multer({ storage: storage });
// Start listening
const port = process.env.PORT || 3000;
server.listen(port, () => {
    console.log(`App listening on port ${host}:${port}`);
});
