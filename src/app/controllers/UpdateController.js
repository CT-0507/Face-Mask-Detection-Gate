const mongoose = require('mongoose');
const Device = require('../../models/devices.js');
const Log = require('../../models/log.js');
const _io = require('../../index').io;

/*
_io.emit("message", data)
*/
deviceImgUrl = 'https://pivietnam.com.vn/uploads/tiny_uploads/Jetson-Nano-Large.jpg'

class UpdateController {
    // update/device
    async addDevice(req, res) {
        var imgUrl
        if (req.body.name == 'Jetson Nano'){
            imgUrl = deviceImgUrl
        }

        const data = new Device({
            _id: new mongoose.Types.ObjectId(),
            name: req.body.name,
            imgUrl: imgUrl || deviceImgUrl,
            ip: req.body.ip
        });
        
        data.save()
            .then((result) => {
                res.status(200).send(result._id.toString())
            })
            .catch((err) => {
                console.log('Fail to update to database');
                console.log(err);
                res.status(400).send('Fail to update device information to db.');
            })
    }
  
   

   

}

module.exports = new UpdateController();
