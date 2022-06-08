// const dashboardRouter = require('./dashboard');
// const mainRouter = require('./main');
// const chartsRouter = require('./charts');
// const logsRouter = require('./logs');
const updateRouter = require('./update.js');
const refreshRouter = require('./refresh.js');
// const ledsRouter = require('./leds.js');
// const loginRouter = require('./login.js');
// const signUpRouter = require('./sign-up.js');
//const signUpRouter = require('./');
const loginRouter = require('./login.js');
const dashboardRouter = require('./dashboard.js');
const logoutRouter = require('./logout.js');
const chartsRouter = require('./charts.js');
function route(app) {
    
    app.use('/dashboard', dashboardRouter);
    app.use('/logout', logoutRouter);
    app.use('/update', updateRouter); // /update
    app.use('/refresh', refreshRouter);
    app.use('/charts', chartsRouter);
    app.use('/', loginRouter);
}

module.exports = route;
