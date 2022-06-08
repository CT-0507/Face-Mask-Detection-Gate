const express = require('express');
const router = express.Router();

const manualController = require('../app/controllers/ManualController.js');

router.post('/', manualController.publish);
module.exports = router;
