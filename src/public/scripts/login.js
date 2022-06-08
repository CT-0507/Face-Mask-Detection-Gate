const container = document.querySelector('.container'),
    pwShowHide = document.querySelectorAll('.showHidePw'),
    pwFields = document.querySelectorAll('.password'),
    signUp = document.querySelector('.signup-link'),
    login = document.querySelector('.login-link');
btnLogin = document.querySelector('.btn-login');
btnSignup = document.querySelector('.btn-signup');
console.log(btnLogin);
console.log(btnSignup);
//   js code to show/hide password and change icon
pwShowHide.forEach((eyeIcon) => {
    eyeIcon.addEventListener('click', () => {
        pwFields.forEach((pwField) => {
            if (pwField.type === 'password') {
                pwField.type = 'text';

                pwShowHide.forEach((icon) => {
                    icon.classList.replace('uil-eye-slash', 'uil-eye');
                });
            } else {
                pwField.type = 'password';

                pwShowHide.forEach((icon) => {
                    icon.classList.replace('uil-eye', 'uil-eye-slash');
                });
            }
        });
    });
});

// js code to appear signup and login form
signUp.addEventListener('click', () => {
    container.classList.add('active');
});
login.addEventListener('click', () => {
    container.classList.remove('active');
});
btnLogin.addEventListener('click', () => {
    //alert('you click login');
    $('.login').submit(function (e) {
        e.preventDefault();
    });
    const username = $('#login-username').val();
    const password = $('#login-password').val();
    fetch('http://localhost:3000/login', {
        // Adding method type
        method: 'POST',

        // Adding body or contents to send
        body: JSON.stringify({
            username: username,
            password: password,
        }),
        // Adding headers to the request
        headers: {
            'Content-type': 'application/json; charset=UTF-8',
        },
    })
        // Converting to JSON
        .then((response) => {
            console.log(response.status);
            if (response.status === 200) {
                console.log('Success:');
                $('.wrong-login').attr('style', 'display:none;');
                location.href = '/dashboard';
            } else {
                console.log('wrong password or username:');
                $('.wrong-login').attr('style', 'display:block;color:red;');
            }
        })
        .then((data) => {
            //console.log('Success:', data);
        })
        .catch((error) => {
            //console.error('Error:', error);
        });
});
btnSignup.addEventListener('click', () => {
    //alert('you click login');
    $('.signup').submit(function (e) {
        e.preventDefault();
    });
    const nickname = $('#nickname').val();
    const username = $('#register-username').val();
    const password = $('#register-password').val();
    fetch('http://localhost:3000/sign-up', {
        // Adding method type
        method: 'POST',

        // Adding body or contents to send
        body: JSON.stringify({
            nickname: nickname,
            username: username,
            password: password,
        }),

        // Adding headers to the request
        headers: {
            'Content-type': 'application/json; charset=UTF-8',
        },
    })
        // Converting to JSON
        .then((response) => {
            console.log(response.status);
            if (response.status === 200) {
                console.log('Success:');
                location.href = '';
            } else {
                console.log('user exist:');
                $('.user-exist').attr('style', 'display:block;color:red;');
            }
        })
        .then((data) => {
            //console.log('Success:', data);
        })
        .catch((error) => {
            //console.error('Error:', error);
        });
});
// document.addEventListener('DOMContentLoaded', function() {
//     const loginForm = $('.login');
//         registerForm = $('.signup');
//         btnLogin = $('.btn-login');
//         btnSignin = $('.btn-signup');
//         console.log(loginForm);
//         console.log(registerForm);
//         console.log(btnLogin);
//         console.log(btnSignin);
//         btnLogin.click(function (e) {
//             e.preventDefault();
//             loginForm.action = '/courses/' + 'restore?_method=PATCH';
//             loginForm.submit();
//         });
// });
