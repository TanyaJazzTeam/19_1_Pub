Erster Header | Zweiter Header
--- | ---
Inhaltszelle | Inhaltszelle
Inhaltszelle | Inhaltszelle

Erster Header | Zweiter Header
--- | ---
Inhaltszelle | Inhaltszelle
Inhaltszelle | Inhaltszelle

![Zitadelle](https://vignette.wikia.nocookie.net/masseffect/images/d/d7/MassEffect2Citadel.jpg/revision/latest?cb=20100721191415)

Linksbündig | Mittig ausgerichtet | Rechtsbündig
:-- | :-: | --:
Spalte 3 ist | ein wortreicher Text | **1600 $**
Spalte 2 ist | zentriert | 12 $
Zebrastreifen | sind ordentlich | ~~$1~~

Dillinger ist ein cloudfähiger, für Mobilgeräte geeigneter Offline-Speicher- und AngularJS-basierter HTML5-Markdown-Editor.

- Geben Sie links etwas Markdown ein
- Siehe HTML rechts
- Magie

# WAHR

- Importieren Sie eine HTML-Datei und beobachten Sie, wie sie auf magische Weise in Markdown konvertiert wird
- Bilder per Drag-and-Drop verschieben (erfordert die Verknüpfung Ihres Dropbox-Kontos)

Du kannst auch:

- Importieren und speichern Sie Dateien von GitHub, Dropbox, Google Drive und One Drive
- Ziehen Sie Markdown- und HTML-Dateien per Drag-and-Drop in Dillinger
- Exportieren Sie Dokumente als Markdown, HTML und PDF

Markdown ist eine leichte Auszeichnungssprache, die auf den Formatierungskonventionen basiert, die Menschen normalerweise in E-Mails verwenden. Wie [John Gruber] auf der [Markdown-Site][df1] schreibt

> Das übergeordnete Designziel für die Formatierungssyntax von Markdown besteht darin, sie so lesbar wie möglich zu machen. Die Idee dahinter ist, dass ein Markdown-formatiertes Dokument so wie es ist, als reiner Text, veröffentlicht werden kann, ohne dass es aussieht, als wäre es mit Tags oder Formatierungsanweisungen versehen worden.

Dieser Text, den Sie hier sehen, ist *tatsächlich* in Markdown geschrieben! Um ein Gefühl für die Syntax von Markdown zu bekommen, geben Sie Text in das linke Fenster ein und sehen Sie sich die Ergebnisse im rechten Fenster an.

### FALSCH

Dillinger nutzt eine Reihe von Open-Source-Projekten, um ordnungsgemäß zu funktionieren:

- [AngularJS] – HTML verbessert für Web-Apps!
- [Ace Editor] – toller webbasierter Texteditor
- [markdown-it] – Markdown-Parser richtig gemacht. Schnell und einfach erweiterbar.
- [Twitter Bootstrap] – großartiges UI-Boilerplate für moderne Web-Apps
- [node.js] – Ereignis-E/A für das Backend
- [Express] – schnelles Node.js-Netzwerk-App-Framework [@tjholowaychuk]
- [Gulp] – das Streaming-Build-System
- [Breakdance](https://breakdance.github.io/breakdance/) – HTML-zu-Markdown-Konverter
- [jQuery] - hm

Und natürlich ist Dillinger selbst Open Source mit einem [öffentlichen Repository][dill] auf GitHub.

### Installation

![Ilos](https://lh3.googleusercontent.com/proxy/DDV8a7sLIWurhJtW8Ego9bq-JlwpfFFoR0tkLJQKKYXEXoWHB6ZUP5jGKD2VcYt3z1QVsgcn6L3GoU1ns8m9fvi3U51GzddA70ZUMHgzHvjl4-i7YOJY9cShBPrfjUhMQhxaJ97WFBp612XmjMXVGypfGkiBarN4PWxhiHkiYYNW7HGbtTpOcyt9GQ4Q23C2noxLTWFXZMcQZhRpQA_qzu2n6_H6CPViBnhSHpEl4JZAPaGCSJqgZg)

Für die Ausführung von Dillinger ist [Node.js](https://nodejs.org/) v4+ erforderlich.

Installieren Sie die Abhängigkeiten und devDependencies und starten Sie den Server.

```sh
$ cd dillinger
$ npm install -d
$ node app
```

Für Produktionsumgebungen...

```sh
$ npm install --production
$ NODE_ENV=production node app
```

### Plugins

Dillinger wird derzeit um folgende Plugins erweitert. Anweisungen zur Verwendung in Ihrer eigenen Anwendung finden Sie unten verlinkt.

Plugin | Liesmich
--- | ---
Dropbox | [plugins/dropbox/README.md][PlDb]
GitHub | [plugins/github/README.md][PlGh]
Google Drive | [plugins/googledrive/README.md][PlGd]
Eine Fahrt | [plugins/onedrive/README.md][PlOd]
Mittel | [plugins/medium/README.md][PlMe]
Google Analytics | [plugins/googleanalytics/README.md][PlGa]

### Entwicklung

## alt

Die Kennzahlen, aus denen sich die Core Web Vitals zusammensetzen, werden sich im Laufe der Zeit [weiterentwickeln](#evolving-web-vitals) . Das aktuelle Set für 2020 konzentriert sich auf drei Aspekte des Benutzererlebnisses – *Laden* , *Interaktivität* und *visuelle Stabilität* – und umfasst die folgenden Metriken

Möchten Sie einen Beitrag leisten? Großartig!

Dillinger verwendet Gulp + Webpack für eine schnelle Entwicklung. Nehmen Sie eine Änderung in Ihrer Datei vor und sehen Sie sofort Ihre Aktualisierungen!

Öffnen Sie Ihr Lieblingsterminal und führen Sie diese Befehle aus.

Erste Registerkarte:

```sh
$ node app
```

Zweiter Tab:

```sh
$ gulp watch
```

(optional) Drittens:

```sh
$ karma test
```

#### Bauen für die Quelle

Zur Produktionsfreigabe:

```sh
$ gulp build --prod
```

Generieren vorgefertigter ZIP-Archive zur Verteilung:

```sh
$ gulp build dist --prod
```

### Docker

Dillinger lässt sich sehr einfach in einem Docker-Container installieren und bereitstellen.

Standardmäßig stellt der Docker Port 8080 bereit. Ändern Sie dies daher bei Bedarf in der Docker-Datei. Wenn Sie fertig sind, verwenden Sie einfach die Docker-Datei, um das Image zu erstellen.

```sh
cd dillinger
docker build -t joemccann/dillinger:${package.json.version} .
```

Dadurch wird das Dillinger-Image erstellt und die erforderlichen Abhängigkeiten eingebunden. Stellen Sie sicher, dass Sie `${package.json.version}` durch die aktuelle Version von Dillinger ersetzen.

Wenn Sie fertig sind, führen Sie das Docker-Image aus und ordnen Sie den Port dem gewünschten Port auf Ihrem Host zu. In diesem Beispiel ordnen wir Port 8000 des Hosts einfach Port 8080 des Docker zu (oder dem Port, der in der Docker-Datei offengelegt wurde):

```sh
docker run -d -p 8000:8080 --restart="always" <youruser>/dillinger:${package.json.version}
```

Überprüfen Sie die Bereitstellung, indem Sie in Ihrem bevorzugten Browser zu Ihrer Serveradresse navigieren.

```sh
127.0.0.1:8000
```

#### Kubernetes + Google Cloud

Siehe [KUBERNETES.md](https://github.com/joemccann/dillinger/blob/master/KUBERNETES.md)

### Todos
