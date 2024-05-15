function uploadAndProcess() {
    var fileInput = document.getElementById('fileInput');
    var file = fileInput.files[0];

    if (!file) {
        alert('Veuillez sélectionner un fichier.');
        return;
    }

    var formData = new FormData();
    formData.append('file', file);
    fetch('https://5fcc-194-5-53-100.ngrok-free.app/mask_image', {
        method: 'POST',
        headers: {
            'accept': 'image/*'
        },
        body: formData
    })
    .then(response => {
        return response.blob();
    })
    .then(blob => {
        var imageUrl = URL.createObjectURL(blob);
        var outputDiv = document.getElementById('output');
        outputDiv.innerHTML = '<img src="' + imageUrl + '">';
    })
    .catch(error => {
        console.error('Error', error);
        alert('Error');
    });
}

function display_cloudWords() {
    var choix = document.querySelector('input[name="choixImage"]:checked').value;
    var imageAffichee = document.getElementById('main-paragraph-image-display');

    switch (choix) {
        case 'image1':
            imageAffichee.src = "../static/images/df_pos.png";
            break;
        case 'image2':
            imageAffichee.src = "../static/images/df_neg.png";
            break;
        case 'image3':
            imageAffichee.src = "../static/images/df_pos.png";
            break;
        default:
            imageAffichee.src = "../static/images/df_pos.png";
    }
}

console.log("Le fichier JavaScript est correctement chargé.");