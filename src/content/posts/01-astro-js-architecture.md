---
title: "두 번째 개발 블로그 시작, Astro.js와 함께"
publishedAt: 2024-02-18
description: "Astro의 아일랜드 아키텍처가 무엇인지, 이에 따른 성능은 어떤지 살펴보았습니다."
slug: "about-astro-island-architecture"
isPublish: true
---

## 새로운 블로그 툴로 Astro.js를 선택한 이유

3년 전 Gatsby를 사용해 블로그를 잠시 운영했던 이후로 처음 블로그를 시작하는거라, 새로운 SSG툴이나 블로그 템플릿으로써 더 활용도 있는 라이브러리가 있는지 궁금해서 이것저것 알아보았다. 그러다 우연히 Gatsby에서 Astro로 블로그를 이전한 후기를 담은 [블로그 글](https://johnny-mh.github.io/post/gatsby-to-astro/)을 접하게 되었고, 더 가볍고 빠르다는 이야기를 보고 흥미를 가지게 되었다. 와중에 맘에드는 블로그 템플릿도 있어서 아일랜드 아키텍처도 공부해볼 겸 블로그를 선택하게 되었다.

## Island Architecture

Astro의 핵심 개념은 아일랜드 아키텍처이다. 아일랜드 아키텍처는 직관적으로 알 수 있듯이 페이지에서 각 interactive한 UI들이 독립적으로 동작하는 개념을 이야기한다 (HTML로 unit이 구성된다는 점에서 micro-frontend와는 다른 개념). 각 UI를 섬으로 나눈 이유는 `hydration`을 효과적으로 하기 위함인데, 기존 SSR과의 동작 차이는 다음 사진과 같다.

<img
  src="https://res.cloudinary.com/ddxwdqwkr/image/upload/f_auto/v1633284886/patterns.dev/theislandsarch--avuxy9rrkk8.png"
  width="800"
  height="400"
/>

> 사진 출처: https://www.patterns.dev/vanilla/islands-architecture/

기존 SSR은 hydration이 한꺼번에 진행이 되는데, 아일랜드 아키텍처는 hydration이 필요한 island 단위의 컴포넌트들이 독립적으로 hydrate 된다. 당연한 이야기겠지만, 이렇게 독립적으로 운영이 되면 한 컴포넌트의 성능 이슈가 다른 컴포넌트에 영향을 미치지 않게 된다.

## Progressive hydration

독립적인 hydration은 병렬적으로 실행이 되는데, 이는 클라이언트에서 `client:*` 라는 client directives를 컴포넌트에 전달해 hydration 우선 순위 조정이 가능하다. 우선 순위가 High, Medium, Low로 5가지의 옵션이 존재하는데, 각각 우선 순위에 따라 동작하는 코드도 조금씩 다르다.

우선 `client:load`로 동작하는 **가장 우선순위가 높은** directive를 살펴보면, 내부 코드는 별 다른 기다림 없이 바로 hydrate를 실행하도록 되어있다. 이는 load 되자마자 바로 hydrate하는 코드로, 빨리 상호작용이 필요한 UI에 쓰인다.
```js
const loadDirective: ClientDirective = async (load) => {
	const hydrate = await load();
	await hydrate();
};

export default loadDirective;
```
> https://github.com/withastro/astro/blob/main/packages/astro/src/runtime/client/load.ts

두번째 **medium** 우선 순위를 가지고 있는 `client:idle`은 `requestIdleCallback`으로 코드 제어 부분이 들어가있다. 이는 initial load가 끝나면 실행이 되고, 즉각적인 interactive 처리는 필요하지 않은 컴포넌트에 적합하다.

```js
const cb = async () => {
	const hydrate = await load();
	await hydrate();
};
if ('requestIdleCallback' in window) {
	(window as any).requestIdleCallback(cb);
} 
```
> https://github.com/withastro/astro/blob/main/packages/astro/src/runtime/client/idle.ts

---

### *requestIdleCallback?*

위 requestIdleCallback을 실무에서 사용해본 적이 없어 잠깐 살펴보았는데, MDN에 나온 정의는 다음과 같다.

> window.requestIdleCallback() 메서드는 브라우저의 idle 상태에 호출될 함수를 대기열에 넣습니다. 이를 통해
> 개발자는 **애니메이션 및 입력 응답과 같은 대기 시간이 중요한 이벤트에 영향을 미치지 않고** 메인 이벤트 루프에서 
> **백그라운드 및 우선 순위가 낮은 작업**을 수행 할 수 있습니다.

모든 작업과 세부 작업이 끝나면 브라우저는 유휴 기간에 들어가게 되는데, 이때 실행되는 함수라고 한다. 따라서 마이크로태스크큐 작업, setTimeout, requestIdleCallback 이렇게 세 가지 함수가 일괄 실행되면, 순서대로 **마이크로태스큐 작업 > setTimeout > requestIdleCallback** 순으로 실행이 된다.

---

마지막으로 우선순위가 낮은 directive들에는 `client:visible`, `client:media`, `client:only`가 있는데, `client:visible`만 잠시 살펴보자면 이는 IntersectionObserver로 유저에게 보여지는 부분만 hydration하도록 조정이 되어있다. 

```js
const cb = async () => {
  const hydrate = await load();
  await hydrate();
};

const io = new IntersectionObserver((entries) => {
  for (const entry of entries) {
    if (!entry.isIntersecting) continue;
    // As soon as we hydrate, disconnect this IntersectionObserver for every `astro-island`
    io.disconnect();
    cb();
    break; // break loop on first match
  }
}, ioOptions);

for (const child of el.children) {
  io.observe(child);
}

```
> https://github.com/withastro/astro/blob/main/packages/astro/src/runtime/client/visible.ts


이렇게 Astro에서는 각각 컴포넌트별로 우선순위를 받아 병렬적으로 hydration처리가 되고 따라서 모든 컴포넌트가 hydration되기 기다리는 것이 아니라 다른 컴포넌트에서 hydration이 일어나는 와중에도 미리 끝난 컴포넌트와는 상호작용이 가능해 FID(First Input Delay)가 빨라지게 된다.

## Performance

이 아일랜드 아키텍처를 살펴보면, hydration을 분할 & 병렬적으로 진행하니 적어도 블로그 같은, 상호작용이 크게 필요하지 않은 웹사이트에서는 기존 SSR보다 이점이 있다는 것을 알게 되었다. 그러나 SSG만 비교한다고 했을 때 Gatsby보다도 더 빠를까? 답은 그렇다이다. Astro 공식 문서에 보면 아까 위에서 설명했던, hydration 우선순위를 조정하는 directive를 아예 넣지 않으면 해당 컴포넌트는 아예 자바스크립트가 없이 렌더된다. 따라서 자바스크립트 파일과 함께 렌더되는 Gatsby보다 빠를 수 밖에 없는 것이다.

Astro측은 Gatsby, Next.js말고도 Nuxt, Remix, SveltKit 등 여러 라이브러리의 Core Web Vitals를 측정해 작성해 둔 글이 있는데, Astro가 모든 지표에서 압도적이라는 결과를 보여주고 있다. [Astro 공식 블로그](https://astro.build/blog/2023-web-framework-performance-report/) 블로그 글에 따르면 앞에서 설명한 First Input Delay 외에도 CLS(Cumulative Layout Shift), LCP(Large Contentful Paint), INP(Interaction to Next Paint) 지표에서도 다른 라이브러리에 비해 지표가 우위에 있다. 

이 외에도 Gatsby에서 Astro로 이사해 Lighthouse 점수가 올라갔다는 다른 개발자의 블로그 후기들이 많이 있었다. 이만하면 블로그로 활용하기에 고민할 필요가 없는 선택지가 아닌가 싶다. 오랜만에 기술 블로그를 시작하려니 뭘 써야 하나 싶었는데 이렇게 첫 글감도 되어주고, 내부 코드도 살펴보고 하니 재밌었다 🙂 좋은 라이브러리 위에서 다시 블로그 생활 시작!

<br/>
<br/>
<br/>

---

### 출처
- [아일랜드 아키텍처] https://docs.astro.build/en
- [아일랜드 아키텍처] https://jasonformat.com/islands-architecture/
- [아일랜드 아키텍처] https://www.patterns.dev/vanilla/islands-architecture/
- [requestIdleCallback] https://intspirit.medium.com/understanding-the-browsers-event-loop-for-building-high-performance-web-applications-part-2-234a78f4a3e7
- [Astro performance] https://astro.build/blog/2023-web-framework-performance-report/