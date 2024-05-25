function display_cloudWords() {
    var choice1 = document.querySelector('input[name="image_choice1"]:checked').value;
    var choice2 = document.querySelector('input[name="image_choice2"]:checked').value;
    var choice3 = document.querySelector('input[name="image_choice3"]:checked').value;

    var displayed_image_1 = document.getElementById('main-paragraph-image-display_1');
    var displayed_image_2 = document.getElementById('main-paragraph-image-display_2');
    var displayed_image_3 = document.getElementById('main-paragraph-image-display_3');
    console.log("ok")

    switch (choice1) {
        case 'image1':
            displayed_image_1.src = "../static/images/df_pos.png";
            break;
        case 'image2':
            displayed_image_1.src = "../static/images/df_neg.png";
            break;
        case 'image3':
            displayed_image_1.src = "../static/images/df_pos.png";
            break;
        default:
            displayed_image_1.src = "../static/images/df_pos.png";
    }

    switch (choice2) {
        case 'image3':
            displayed_image_2.src = "../static/images/countX_pos.png";
            break;
        case 'image4':
            displayed_image_2.src = "../static/images/countX_neg.png";
            break;
        default:
            displayed_image_2.src = "../static/images/countX_pos.png";
    }

    switch (choice3) {
        case 'image5':
            displayed_image_3.src = "../static/images/bert_ROC.png";
            break;
        case 'image6':
            displayed_image_3.src = "../static/images/roberta_ROC.png";
            break;
        default:
            displayed_image_3.src = "../static/images/bert_ROC.png";
    }
}


function predict_sentiment() {
    var selectElement = document.getElementById("options");
    var selectedValue = selectElement.value;

    fetch('http://localhost:8080/predict_sentiment', {
        method: 'POST',
        headers: {
            'Accept': 'application/json',
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ text: selectedValue })
    })
    .then(response => response.json())
    .then(data => {
        let sentiment;
        if (data[0].label == 'LABEL_1') {
            sentiment = "POSITIF";
        } else {
            sentiment = "NEGATIF";
        }
        let score = (data[0].score * 100).toFixed(1);
        var resultDiv = document.getElementById('result');
        resultDiv.innerHTML = `<div>Le sentiment du tweet est <strong class="sentiment">${sentiment}</strong> avec une certitude de <strong class="sentiment">${score}%</strong></div>`;
        console.log(data[0].label, data[0].score);
    })
    .catch(error => {
        console.error('Erreur lors de la requête :', error);
    });
}


console.log("yea")