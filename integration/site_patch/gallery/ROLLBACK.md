Работы (история): /var/www/sonya/gallery/{gallery.js,gallery.css}; app-shell.js: gallery kind 'soon'->'view'; ?v=20261113-gallery in index/opencut/streamer-mode.html.
Откат: cd /var/www/sonya; cp app-shell.js.bak-pre-gallery-* app-shell.js (самый ранний, ls -t | tail -1); for f in index opencut streamer-mode; do cp $f.html.bak-pre-gallery-20261113 $f.html; done
Resume: cp /var/www/sonya/app.js.bak-pre-resume-* (earliest) app.js; sed index.html v=20261113-dl
