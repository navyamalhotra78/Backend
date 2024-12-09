const express = require('express');
const nodemailer = require('nodemailer');
const bodyParser = require('body-parser');
const app = express();

app.use(bodyParser.json());

// Setup Nodemailer transport
const transporter = nodemailer.createTransport({
  service: 'gmail',  // Change to your email service
  auth: {
    user: 'navy0807@gmail.com',  // Your email address
    pass: 'imco rsxd ultc wuqm'    // Your email password or app-specific password
  }
});

// Email route
app.post('/send-email', (req, res) => {
  const { subject, text, to } = req.body;

  const mailOptions = {
    from: 'navya0807@gmail.com',
    to,
    subject,
    text
  };

  transporter.sendMail(mailOptions, (error, info) => {
    if (error) {
      console.log(error);
      return res.status(500).send({ message: 'Error sending email' });
    }
    console.log('Email sent: ' + info.response);
    res.status(200).send({ message: 'Email sent successfully' });
  });
});

app.listen(3000, () => {
  console.log('Email service listening on port 3000');
});
