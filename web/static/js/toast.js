function popToast(msg){
  var snackbarContainer = document.querySelector('#toast');
  var data = {message: msg, timeout: 4000};
  snackbarContainer.MaterialSnackbar.showSnackbar(data);
}