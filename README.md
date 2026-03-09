# 2026-Spring

## 如何提交一个 pull request (pr)？

有时候你需要把你的代码或者经你修改后的代码提交到这个仓库，应该怎么做？

<img align="right" width="300" src="https://firstcontributions.github.io/assets/Readme/fork.png" alt="复制此仓库代码" />


## Fork（复制）本代码仓库

点击图示中的按钮去 Fork 这个代码仓库。
这个操作会将代码仓库复制到你的账户名下。

## Clone（克隆）代码仓库

<img align="right" width="300" src="https://firstcontributions.github.io/assets/Readme/clone.png" alt="克隆此仓库代码" />

接下来将复制的代码仓库克隆到你的电脑上。点击图示中的绿色按钮，接着点击复制到剪切板按钮（将代码仓库地址复制下来）

随后打开命令行窗口，敲入如下 git 命令：

```
git clone "刚才复制的 url 链接"
```
"刚才复制的 url 链接"（去掉双引号）就是复制到你账户名下的代码仓库地址。获取该链接的方法详见上一步。

<img align="right" width="300" src="https://firstcontributions.github.io/assets/Readme/copy-to-clipboard.png" alt="将url链接复制到剪贴板" />

譬如：
```bash
git clone git@github.com:<Github用户名>/first-contributions.git
```

'Github 用户名' 指的是你的 Github 用户名。这一步，这个操作将会克隆你账户名下 first-contributions 这个代码仓库到本地电脑上。

## 新建一个代码分支

在命令行窗口中把目录切换到 first-contributions

```bash
cd first-contributions
```
接下来使用 `git switch` 命令新建一个代码分支
```bash
git switch -c <新分支的名称>
```

譬如：
```bash
git switch -c add-myname
```

(新分支的名称不一定需要有 *add*。然而，在新分支的名称加入 *add* 是一件合理的事情，因为这个分支的目的是将你的名字添加到列表中。)

## 对代码进行修改，然后 Commit (提交) 修改

打开 `Contributors.md` 这个文件，更新文件内容，将你的名字加上去，保存修改。`git status` 这命令会列出被改动的文件。接着 `git add` 这命令则可以添加你的改动，就像如下这条命令。

<img align="right" width="450" src="https://firstcontributions.github.io/assets/Readme/git-status.png" alt="修改`Contributors.md`后的git状态" />

```bash
git add Contributors.md
```

现在就可以使用 `git commit` 命令 commit 你的修改了。
```bash
git commit -m "Add <你的名字> to Contributors list"
```
将 `<你的名字>` 替换成你的名字

## 将改动 Push（推送）到 GitHub

使用 `git push` 命令推送代码
```bash
git push origin <分支的名称>
```
将 `<分支的名称>` 替换为之前新建的分支名称。

<details>
<summary> <strong>如果在 push（发布）过程中出 error（错误），点击这里</strong> </summary>

- ### Authentication Error
     <pre>remote: Support for password authentication was removed on August 13, 2021. Please use a personal access token instead.
  remote: Please see https://github.blog/2020-12-15-token-authentication-requirements-for-git-operations/ for more information.
  fatal: Authentication failed for 'https://github.com/<your-username>/first-contributions.git/'</pre>
  去 [GitHub's tutorial](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/adding-a-new-ssh-key-to-your-github-account) 学习如何生成新的 SSH 密匙以及配置。

</details>

## 提出 Pull Request 将你的修改供他人审阅

前往你的 Github 代码仓库，你会看到一个 `Compare & pull request` 的按钮。点击该按钮。

<img style="float: right;" src="https://firstcontributions.github.io/assets/Readme/compare-and-pull.png" alt="创建 pull request" />

接着再点击 `Create pull request` 按钮，正式提交 pull request。

<img style="float: right;" src="https://firstcontributions.github.io/assets/Readme/submit-pull-request.png" alt="提交 pull request" />

不久之后，我便会把你所有的变化合并到这个项目的主分支。更改合并后，你会收到一封电子邮件通知。

### *由于你也是这个项目的 collaborator，所以你提交的代码无需经过我的审核，可以由你自己直接 merge*

## 如果我提交了一些更改，如何让你的本地与GitHub上的内容一致？

执行
```bash
# 下载远程仓库最新的提交记录，不会修改你的工作区。
git fetch --all
# 把当前分支强制重置到 origin/main 的状态。
git reset --hard origin/main
# 删除未被 Git 跟踪的文件和目录
git clean -fd
```

我会尽量不删除你的任何更改，可以放心地 reset