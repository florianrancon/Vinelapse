#Recalage des noms d'images générées par after effect sur les noms vinelapse originaux avec date

img_folder="C://Users//Utilisateur//Documents//IMS//VINELAPSE//IMAGES_TRI//fixcam10//"
AE_folder="C://Users//Utilisateur//Documents//IMS//VINELAPSE//GRAPPES//fixcam10_AME//"


list_img <- list.files(img_folder,pattern = "*.jpg")
list_GT <- list.files(AE_folder,pattern = "*png")

#Vérifier que les deux dossiers ont le même nombre d'images
if (length(list_img) == length(list_GT)) {
  print("check nombre images ok")
  for (i in 1:length(list_GT)) {
    nom_masque=paste0("mask_",substring(list_img[i],6,9),".png")
    if (nom_masque != list_GT[i] ) {
      print(paste0("renaming mask file to ",nom_masque))
      file.rename(paste(AE_folder,list_GT[i],sep="/"), paste(AE_folder,nom_masque,sep="/"))
    }
  }
}
print("done")