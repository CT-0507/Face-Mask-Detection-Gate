const express = require('express');
const router = express.Router();
var multer = require('multer');

var storage = multer.diskStorage({
    destination: (req, file, cb) => {
        cb(null, 'src/uploads');
    },
    filename: (req, file, cb) => {
        cb(null, file.fieldname + '-' + Date.now() + '.png');
    },
});

var upload = multer({ storage: storage });
const enterLogController = require('../app/controllers/EnterLogController');

// router.delete('/:id', enterLogController.destroy);
// router.get('/filter', enterLogController.filter);
// router.get('/filter/:page', enterLogController.filter);
router.get('/filter', enterLogController.filter);
router.get('/filter/:page', enterLogController.filter);
router.get('/detail/:id', enterLogController.detail);
router.delete('/delete-all-log', enterLogController.destroyAll);
router.delete('/:id', enterLogController.destroy);
router.post('/upload', upload.single('image'), enterLogController.upload);
router.get('/', enterLogController.index);
// router.get('/filter', logController.filter);
// router.delete('/', logController.destroy);
// router.get('/', logController.index);

module.exports = router;
