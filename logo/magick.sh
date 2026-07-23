magick sheaf-logo.png -resize 512x512 -background none -gravity center -extent 512x512 sheaf-dark.png
magick sheaf-logo.png -resize 64x64 -background none -gravity center -extent 64x64 favicon.png
magick sheaf-dark.png -fill white -colorize 100% sheaf-light.png

DEST="../docs/images/"
for img in sheaf-dark.png sheaf-light.png favicon.png; do
	cp $img $DEST
done



