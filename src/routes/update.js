const express = require('express');
const router = express.Router();

const updateController = require('../app/controllers/UpdateController');

router.post('/device', updateController.addDevice); // /update/device with json file
module.exports = router;
