"use strict";
document.querySelectorAll("form[data-decision]").forEach((form) => {
  const select = form.querySelector("[name=action]");
  const classification = form.querySelector("[name=override_classification]");
  const reason = form.querySelector("[name=override_reason]");
  const update = () => {
    const override = select.value === "override";
    classification.required = override;
    reason.required = override;
    classification.disabled = !override;
    reason.disabled = !override;
  };
  select.addEventListener("change", update);
  update();
});
