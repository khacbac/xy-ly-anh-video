# AvatarShield — ETHICS

> Project: IVP501 final project, branch `ivp-pure` (spec v1.0).
> Last updated: 2026-06-02.

This document records the ethical framing of the project and the
non-claims that any reader (instructor, reviewer, downstream user) should
be aware of before treating any artifact in this repository as a privacy
product.

---

## 1. Research status

AvatarShield is a **course research prototype** built for FSB-MSE
IVP501. It is not a deployable child-safety product, not a commercial
service, and not a regulatory-grade anonymization tool. Every claim
in the report is bounded by the specification (`spec.md` §4.3
non-claims).

The pure-IVP501 path (this branch) intentionally trades visual quality
for explainability: every transform maps to a specific course slide
(S02–S10), so the report defends the pipeline as a classical
image-processing artifact rather than a wrapper around third-party
models.

## 2. Threat model — what we defend against

In scope (the cases the report measures):

- **Casual / opportunistic face matching** at upload time on social
  platforms — the pipeline lowers the high-frequency texture that
  off-the-shelf embedders (e.g. FaceNet) rely on, as measured by the
  HF-ratio and re-id baselines in §7 of `spec.md`.
- **Aesthetic anonymization** for guardians who want to share clips of
  minors without surrendering an unaltered face image — the cel-shade
  + theme palette gives a consistent stylised look across clips.

Out of scope (the cases the report does **not** defend against):

- **Motivated attacker with reference photos / video.** A persistent
  adversary with un-stylized priors can re-identify the subject via
  pose, hair geometry, voice, background, social graph — none of which
  the pipeline modifies.
- **Re-identification by humans who already know the subject.** The
  whole point of preserving identity through the theme palette is that
  the subject's family / classmates can still recognise them. That same
  signal is available to any other acquaintance.
- **Reversible anonymization.** No effort is made to make the transform
  invertible or to prove non-invertibility under attack.
- **Biometric matching evasion by adversarially trained models.**
  Models that include cel-shade artefacts in their training distribution
  may match through the stylisation. The baselines in `RESULTS.md`
  only cover the FaceNet-vggface2 release.

## 3. Non-claims

The repository explicitly does **not** claim:

- Production-ready or commercial-grade privacy.
- Compliance with COPPA, GDPR, CCPA, or any other regulatory regime.
- Verifiable identity erasure under any threat model stronger than §2.
- Visual fidelity comparable to GAN stylisation (VToonify / AnimeGAN);
  the pipeline is intentionally less polished in exchange for full
  pedagogical explainability.
- Self-recognition for strangers — only the subject and people who
  already know them are expected to recognise the stylised output.

These bullets mirror `spec.md` §4.3 — keep them in sync.

## 4. Data / consent

- **No minors as study participants.** The optional self-recognition
  study (`spec.md` §7.4) is restricted to adult team members and adult
  volunteers.
- **Test media.** `samples/input/` is gitignored. Any clip used in
  development or in the demo must be either authored by the team, used
  with documented consent from the subject, or sourced from a public
  domain / Creative Commons collection with attribution.
- **No avatar PNGs / VRoid exports.** The legacy path stored a per-user
  avatar image; v1.0 stores only numeric Lab triples in
  `assets/themes/*.json`. Themes therefore carry no biometric or
  character-identity information.

## 5. Dual-use disclosure

Classical anonymization at this level has a smaller misuse surface than
the legacy VToonify / face-swap path:

- No identity *impersonation*. There is no avatar paste-on, no face
  swap, no generative head model. The transform only re-colours and
  cel-shades the existing head region.
- No model weights ship with the repository — neither the legacy
  VToonify / BiSeNet checkpoints nor any new model. The pipeline runs
  with `opencv-python`, `numpy`, and `Pillow` only (`requirements.txt`,
  3 lines).
- No third-party character IP. The five starter themes are hand-authored
  Lab statistics and do not derive from any commercial illustration,
  character, or franchise.

Risks that remain:

- A user could pair the pipeline with a separately collected dataset of
  child faces. The repository does not provide such a dataset and
  documents in §4 that test media must be consented.
- A user could disable the head-region mask and apply the cel-shade to
  arbitrary footage. The output remains a non-photorealistic stylised
  video, with no identity transfer capability.

## 6. Reproducibility & audit

- The pipeline is deterministic apart from K-means (cluster centres are
  EMA-smoothed across frames, so per-clip jitter is bounded by the
  initial frame's K-means seed). Re-running on the same input clip
  produces visually equivalent output.
- Evaluation scripts under `scripts/eval_*.py` regenerate every number
  in `RESULTS.md` from the source clip and the five baseline renders.
  See `RESULTS.md` for the command list.
- Legacy specs (`spec_legacy.md`, `plan_legacy.md`) are kept in-tree
  for auditing the project's design history.

## 7. License & attribution

Code is released under MIT (or equivalent permissive license — confirm
at submission). Third-party citations live in `spec.md` §14
(Viola–Jones cascade, Reinhard color transfer, XDoG line art,
1€ filter). FaceNet weights (vggface2) are only used at evaluation
time via `facenet-pytorch`; they are not shipped with the repository.

## 8. Reporting concerns

If a reader spots an ethical issue, a non-disclosed limitation, or a
factual inaccuracy in the report or this document, please open an
issue on the repository or contact the team via the channels listed in
the course submission. Updates land here with a dated entry rather
than a silent edit.
