// Calcule le score avec les paramètres du modèle exporté par train_model.py (model.js)
const $ = id => document.getElementById(id);

const TEXTES = {
  ratio_mensualite_revenu: { up: "Mensualité élevée par rapport au revenu", down: "Mensualité raisonnable par rapport au revenu",
    conseil: "Réduire le montant ou allonger la durée pour alléger la mensualité." },
  retards_precedents: { up: "Retards de paiement dans l’historique", down: "Peu ou pas de retards de paiement",
    conseil: "Examiner l’historique de remboursement avant de décider." },
  anciennete_activite: { up: "Activité récente", down: "Activité ancienne et stable",
    conseil: "Demander des justificatifs de l’activité (stabilité des revenus)." },
  ratio_epargne_revenu: { up: "Épargne faible par rapport au revenu", down: "Épargne confortable",
    conseil: "Encourager la constitution d’une épargne de garantie." },
  a_garant: { up: "Absence de garant", down: "Présence d’un garant",
    conseil: "Proposer la présence d’un garant." },
  age: { up: "Profil d’âge moins favorable", down: "Profil d’âge favorable" },
  nb_credits_precedents: { up: "Nombre de crédits antérieurs", down: "Expérience de crédits antérieurs" },
};

const nf = n => Math.round(n).toLocaleString("fr-FR");

function lire() {
  const revenu = +$("revenu").value, duree = +$("duree").value;
  return {
    age: +$("age").value,
    anciennete_activite: +$("anciennete").value,
    ratio_mensualite_revenu: revenu > 0 ? (+$("montant").value / duree) / revenu : 0,
    ratio_epargne_revenu: revenu > 0 ? Math.min(Math.max(+$("epargne").value / revenu, 0), 3) : 0,
    nb_credits_precedents: +$("nbcredits").value,
    retards_precedents: +$("retards").value,
    a_garant: $("garant").checked ? 1 : 0,
    revenu, mensualite: revenu > 0 ? +$("montant").value / duree : 0,
  };
}

function calculer() {
  const x = lire();
  let z = MODEL.intercept;
  const contrib = MODEL.features.map((f, i) => {
    const c = MODEL.coef[i] * (x[f] - MODEL.mean[i]) / MODEL.scale[i];
    z += c;
    return { f, c };
  });
  const p = 1 / (1 + Math.exp(-z));
  const score = Math.round((1 - p) * 100);
  const niveau = p < 0.10 ? ["Risque faible", "var(--ok)"] : p < 0.20 ? ["Risque moyen", "var(--mid)"] : ["Risque élevé", "var(--bad)"];

  $("placeholder").hidden = true; $("out").hidden = false;
  $("scoreVal").textContent = score;
  $("gaugeBar").style.width = score + "%"; $("gaugeBar").style.background = niveau[1];
  $("level").textContent = niveau[0]; $("level").style.color = niveau[1];
  $("prob").textContent = `Probabilité estimée de défaut : ${(p * 100).toFixed(1).replace(".", ",")} % · mensualité estimée : ${nf(x.mensualite)} FCFA (${(x.mensualite / x.revenu * 100).toFixed(0)} % du revenu)`;

  const ups = contrib.filter(o => o.c > 0.05).sort((a, b) => b.c - a.c);
  const downs = contrib.filter(o => o.c < -0.05).sort((a, b) => a.c - b.c);
  const li = arr => arr.length ? arr.map(t => `<li>${t}</li>`).join("") : "<li>Aucun facteur marquant</li>";
  $("up").innerHTML = li(ups.slice(0, 3).map(o => TEXTES[o.f].up));
  $("down").innerHTML = li(downs.slice(0, 3).map(o => TEXTES[o.f].down));
  const conseils = ups.map(o => TEXTES[o.f].conseil).filter(Boolean).slice(0, 3);
  $("reco").innerHTML = conseils.length ? conseils.map(t => `<li>${t}</li>`).join("") : "<li>Dossier solide : poursuivre l’instruction normale.</li>";
}

function exemple() {
  Object.entries({ age: 24, revenu: 70000, anciennete: 0.5, epargne: 0, montant: 600000, nbcredits: 3, retards: 2 }).forEach(([k, v]) => $(k).value = v);
  $("duree").value = "6"; $("garant").checked = false; calculer();
}

$("calc").addEventListener("click", calculer);
$("ex").addEventListener("click", exemple);
