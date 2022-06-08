const express = require('express');
const router = express.Router();

const loginController = require('../app/controllers/LoginController');

router.post('/sign-up', loginController.signup);
router.post('/login', loginController.login);
router.get('/', loginController.index);

module.exports = router;
